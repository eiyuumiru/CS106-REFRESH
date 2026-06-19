from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk


def _ensure_punkt() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def split_sentences(text: str, max_sents: int) -> list[str]:
    _ensure_punkt()
    from nltk.tokenize import sent_tokenize

    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sents[:max_sents]


def tokenize_sentence(sentence: str, max_words: int) -> list[str]:
    _ensure_punkt()
    from nltk.tokenize import word_tokenize

    return word_tokenize(sentence.lower())[:max_words]


class SentenceEncoderCNN(nn.Module):
    def __init__(self, emb_dim: int, sent_emb: int, num_kernels: int):
        super().__init__()
        ch = sent_emb // num_kernels
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, ch, kernel_size=w) for w in range(1, num_kernels + 1)]
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return torch.cat([F.relu(c(x)).max(dim=2).values for c in self.convs], dim=1)


class REFRESHModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        sent_emb: int,
        lstm_hid: int,
        num_kernels: int,
        pad_id: int = 0,
        pretrained_emb=None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.sent_enc = SentenceEncoderCNN(emb_dim, sent_emb, num_kernels)
        self.doc_lstm = nn.LSTM(sent_emb, lstm_hid, batch_first=True)
        self.ext_lstm = nn.LSTM(sent_emb, lstm_hid, batch_first=True)
        self.classifier = nn.Linear(lstm_hid, 2)

    def forward(self, docs):
        B, S, T = docs.shape
        mask = (docs != self.pad_id).any(dim=-1)

        x = self.embedding(docs.view(B * S, T))
        s = self.sent_enc(x).view(B, S, -1)

        _, (h_doc, c_doc) = self.doc_lstm(torch.flip(s, [1]))

        ext_out, _ = self.ext_lstm(s, (h_doc, c_doc))
        logits = self.classifier(ext_out)
        return logits, mask


class RefreshSummarizer:
    def __init__(self, model: REFRESHModel, word2id: dict, meta: dict, device):
        self.model = model
        self.word2id = word2id
        self.meta = meta
        self.device = device
        self.pad_id = meta.get("pad_id", 0)
        self.unk_id = meta.get("unk_id", 1)
        self.max_doc_len = meta.get("max_doc_len", 120)
        self.max_sent_len = meta.get("max_sent_len", 100)
        self.default_m = meta.get("m", 3)

    def _encode(self, text: str):
        raw_sents = split_sentences(text, self.max_doc_len)
        if not raw_sents:
            return None, None, []
        tok = [tokenize_sentence(s, self.max_sent_len) for s in raw_sents]
        doc = torch.zeros(1, self.max_doc_len, self.max_sent_len, dtype=torch.long)
        weight = torch.zeros(1, self.max_doc_len)
        for i, sent in enumerate(tok):
            ids = [self.word2id.get(w, self.unk_id) for w in sent]
            if ids:
                doc[0, i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            weight[0, i] = 1.0
        return doc.to(self.device), weight.to(self.device), raw_sents

    @torch.no_grad()
    def sentence_scores(self, text: str):
        doc, weight, raw_sents = self._encode(text)
        if not raw_sents:
            return [], []
        self.model.eval()
        logits, _ = self.model(doc)
        probs = F.softmax(logits.float(), dim=-1)[0, :, 0] * weight[0]
        scores = probs[: len(raw_sents)].cpu().tolist()
        return raw_sents, scores

    def summarize(self, text: str, m: int | None = None):
        m = m or self.default_m
        raw_sents, scores = self.sentence_scores(text)
        if not raw_sents:
            return [], [], []
        m = min(m, len(raw_sents))
        chosen = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:m])
        summary = [raw_sents[i] for i in chosen]
        return summary, chosen, scores


def lead_summary(text: str, m: int, max_doc_len: int = 120) -> tuple[list[str], list[int]]:
    raw_sents = split_sentences(text, max_doc_len)
    m = min(m, len(raw_sents))
    return raw_sents[:m], list(range(m))


def model_files_status(model_dir: str) -> dict:
    need = {
        "refresh_best.pt": os.path.join(model_dir, "refresh_best.pt"),
        "vocab.json": os.path.join(model_dir, "vocab.json"),
        "meta.json": os.path.join(model_dir, "meta.json"),
    }
    return {name: os.path.exists(path) for name, path in need.items()}


@lru_cache(maxsize=2)
def load_summarizer(model_dir: str, device_str: str = "cpu") -> RefreshSummarizer:
    status = model_files_status(model_dir)
    missing = [n for n, ok in status.items() if not ok]
    if missing:
        raise FileNotFoundError(
            "Thiếu file checkpoint trong '%s': %s" % (model_dir, ", ".join(missing))
        )

    device = torch.device(device_str)
    with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    word2id = {w: i for i, w in enumerate(vocab)}

    model = REFRESHModel(
        vocab_size=meta["vocab_size"],
        emb_dim=meta["emb_dim"],
        sent_emb=meta["sent_emb"],
        lstm_hid=meta["lstm_hid"],
        num_kernels=meta["num_kernels"],
        pad_id=meta.get("pad_id", 0),
    ).to(device)
    state = torch.load(os.path.join(model_dir, "refresh_best.pt"), map_location=device)
    if isinstance(state, dict) and "model" in state and "embedding.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return RefreshSummarizer(model, word2id, meta, device)
