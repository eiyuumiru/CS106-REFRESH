import html
import os

import pandas as pd
import streamlit as st

import refresh_model as rm

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")

st.set_page_config(page_title="REFRESH - Tóm tắt trích xuất", layout="wide")

ORANGE = "#E0551A"
BLUE = "#5B7FA6"

st.markdown(
    f"""
    <style>
      .stApp h1 {{ color:{ORANGE}; }}
      .pick {{ background:{ORANGE}22; border-left:4px solid {ORANGE};
               padding:2px 6px; border-radius:3px; }}
      .lead {{ background:{BLUE}22; border-left:4px solid {BLUE};
               padding:2px 6px; border-radius:3px; }}
      .sent {{ padding:2px 6px; margin:2px 0; display:block; line-height:1.5; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("REFRESH - Tóm tắt văn bản trích xuất")
st.caption(
    "Ranking Sentences for Extractive Summarization with Reinforcement Learning "
    "(Narayan et al., NAACL 2018) - CS106, Nhóm 3"
)


@st.cache_resource(show_spinner="Đang nạp checkpoint REFRESH...")
def get_summarizer():
    return rm.load_summarizer(MODEL_DIR, "cpu")


status = rm.model_files_status(MODEL_DIR)
all_ready = all(status.values())

summarizer = None
load_err = None
if all_ready:
    try:
        summarizer = get_summarizer()
    except Exception as e:
        load_err = str(e)

default_m = summarizer.default_m if summarizer is not None else 3
max_doc_len = summarizer.max_doc_len if summarizer is not None else 120


with st.sidebar:
    st.header("Tuỳ chọn")
    if summarizer is not None:
        st.success("Đã nạp mô hình REFRESH")
    else:
        st.warning("Chưa có checkpoint - chỉ chạy được baseline LEAD.")
        with st.expander("Cách thêm checkpoint", expanded=False):
            st.markdown(
                "Đặt 3 file vào `app/model/`: `refresh_best.pt`, `vocab.json`, `meta.json` "
                "(xuất từ cell EXPORT FOR APP của notebook)."
            )
            for name, ok in status.items():
                st.write(("[OK]    " if ok else "[thiếu] ") + name)
        if load_err:
            st.error(load_err)

    if st.button("Nạp lại checkpoint"):
        get_summarizer.clear()
        try:
            rm.load_summarizer.cache_clear()
        except Exception:
            pass
        st.rerun()

    m = st.slider("Số câu tóm tắt (m)", 1, 8, value=int(default_m))
    show_scores = st.checkbox("Hiện điểm từng câu", value=True)


def highlight(raw_sents, chosen, css):
    chosen = set(chosen)
    parts = []
    for i, s in enumerate(raw_sents):
        cls = css if i in chosen else "sent"
        parts.append(f"<span class='sent'><span class='{cls}'>{html.escape(s)}</span></span>")
    return "<div>" + "".join(parts) + "</div>"


@st.cache_resource(show_spinner=False)
def get_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
    )


def rouge_row(summary_sents, reference):
    s = get_scorer().score(reference.strip(), "\n".join(summary_sents))
    return {
        "ROUGE-1": s["rouge1"].fmeasure * 100,
        "ROUGE-2": s["rouge2"].fmeasure * 100,
        "ROUGE-L": s["rougeLsum"].fmeasure * 100,
    }


text = st.text_area(
    "Văn bản đầu vào (bài tin tức tiếng Anh)",
    height=240,
    placeholder="Paste a news article here...",
)

reference = st.text_area(
    "Gold summary (bắt buộc, để chấm ROUGE)",
    height=140,
    placeholder="Reference summary (required)...",
)

go = st.button("Tóm tắt", type="primary", use_container_width=True)


if go:
    if not text.strip():
        st.error("Vui lòng nhập văn bản.")
        st.stop()
    if not reference.strip():
        st.error("Vui lòng nhập gold summary (bắt buộc để chấm ROUGE).")
        st.stop()

    raw_sents = rm.split_sentences(text, max_doc_len)
    if len(raw_sents) < 1:
        st.error("Không tách được câu nào.")
        st.stop()

    lead_sents, lead_idx = rm.lead_summary(text, m, max_doc_len)
    refresh_sents, refresh_idx, scores = ([], [], [])
    if summarizer is not None:
        refresh_sents, refresh_idx, scores = summarizer.summarize(text, m)

    st.divider()
    st.subheader("Kết quả")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### REFRESH")
        if summarizer is not None:
            st.markdown(" ".join(refresh_sents) or "-")
            if set(refresh_idx) == set(lead_idx):
                st.caption("REFRESH chọn trùng LEAD (bài theo cấu trúc inverted pyramid).")
        else:
            st.info("Chưa có checkpoint REFRESH.")
    with c2:
        st.markdown(f"#### LEAD (baseline, {m} câu đầu)")
        st.markdown(" ".join(lead_sents) or "-")

    st.markdown("##### Câu được chọn (tô màu trên văn bản gốc)")
    hc1, hc2 = st.columns(2)
    with hc1:
        st.caption("REFRESH")
        idx = refresh_idx if summarizer is not None else []
        st.markdown(highlight(raw_sents, idx, "pick"), unsafe_allow_html=True)
    with hc2:
        st.caption("LEAD")
        st.markdown(highlight(raw_sents, lead_idx, "lead"), unsafe_allow_html=True)

    if summarizer is not None and show_scores and scores:
        st.markdown("##### Điểm p(extract) cho từng câu")
        df = pd.DataFrame(
            {
                "Câu": [str(i + 1) for i in range(len(scores))],
                "p(extract)": [round(s, 4) for s in scores],
                "Chọn": ["x" if i in set(refresh_idx) else "" for i in range(len(scores))],
                "Nội dung": [s[:90] for s in raw_sents],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    if reference.strip():
        st.markdown("##### Đánh giá ROUGE so với gold summary")
        rows, names = [], []
        if summarizer is not None and refresh_sents:
            rows.append(rouge_row(refresh_sents, reference)); names.append("REFRESH")
        rows.append(rouge_row(lead_sents, reference)); names.append("LEAD")
        st.table(pd.DataFrame(rows, index=names).round(2))
        if len(rows) == 2:
            r, l = rows[0], rows[1]
            mc = st.columns(3)
            for col, k in zip(mc, ["ROUGE-1", "ROUGE-2", "ROUGE-L"]):
                col.metric(f"{k} (REFRESH)", f"{r[k]:.2f}", f"{r[k] - l[k]:+.2f} vs LEAD")
