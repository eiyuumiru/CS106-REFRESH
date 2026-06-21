# REFRESH — Extractive Summarization Demo

A small web app that runs **extractive text summarization** with a trained
**REFRESH** model (Narayan et al., *Ranking Sentences for Extractive Summarization
with Reinforcement Learning*, NAACL 2018). Paste an English article, get the
top-ranked sentences, and compare against the **LEAD** baseline with **ROUGE** scoring.

The model ranks every sentence with `P(extract)` and picks the top *m*. Inference
runs on CPU — the trained checkpoint is self-contained (embeddings included), so no
GloVe or GPU is needed to serve it.

## Quickstart

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\streamlit run app.py
```

Open http://localhost:8501. The trained checkpoint ships in `model/`, so the app
works out of the box.

> Torch is installed separately from the CPU wheel index to avoid pulling the
> multi-GB CUDA build.

## Usage

1. Paste an English article into the input box.
2. Paste its reference (gold) summary — required, it's what ROUGE scores against.
3. Hit **Tóm tắt**.

You get the REFRESH summary vs. the LEAD baseline, the chosen sentences highlighted
inline, the per-sentence `P(extract)` table, and ROUGE-1/2/L for both.

## Layout

```
app.py            Streamlit UI
refresh_model.py  model architecture, tokenizer, inference
requirements.txt
model/            trained checkpoint + training notebook
sample_texts/     example articles + gold summaries
```

The `model/` directory holds the served checkpoint (`refresh_best.pt`,
`vocab.json`, `meta.json`) plus `REFRESH_pipeline.ipynb` — the notebook that trains
the model and re-exports those three files. Run it end-to-end to reproduce or retrain.

## Notes

- `refresh_model.py` rebuilds the exact architecture from training (CNN sentence
  encoder, kernels 1–7 × 50 → 350; document LSTM-600 reading reversed; extractor
  LSTM seeded from the document state → linear-2 classifier), so the checkpoint
  loads cleanly.
- ROUGE uses `rougeLsum` + Porter stemmer, matching the `pyrouge -m` setup the
  paper reports.
