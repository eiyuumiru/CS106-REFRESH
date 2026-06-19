# REFRESH — Streamlit demo (CS106, Nhóm 3)

Web demo cho paper **Ranking Sentences for Extractive Summarization with Reinforcement Learning** (Narayan et al., NAACL 2018).
Dán một đoạn văn bản tiếng Anh + gold summary -> app sinh **bản tóm tắt trích xuất** bằng mô hình **REFRESH** đã train, so sánh với baseline **LEAD**, và chấm **ROUGE** so với gold summary.

## 1. Cài đặt (BẮT BUỘC dùng venv — không cài vào Python tổng)

```bat
cd app
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Lấy checkpoint REFRESH

App cần 3 file trong `app/model/`: `refresh_best.pt`, `vocab.json`, `meta.json`.

1. Mở `thuc_nghiem/REFRESH_train_for_app.ipynb` trên **Google Colab** (L4 GPU) -> `Runtime -> Run all`.
2. Cell cuối **EXPORT FOR APP** sẽ tạo & tải về `refresh_app_bundle.zip`.
3. Giải nén, copy 3 file vào `app/model/`.

> Chưa có checkpoint, app vẫn chạy được — chỉ hiển thị baseline **LEAD** để test giao diện.

## 3. Chạy

```bat
.venv\Scripts\streamlit run streamlit_app.py
```
Mở trình duyệt tại `http://localhost:8501`.

## 4. Dùng cho video demo

- Dán một bài tin tức tiếng Anh vào ô đầu vào (có thể copy nội dung từ các file trong `sample_texts/`).
- Bấm **Tóm tắt** -> xem câu được tô màu, điểm `P(extract)` từng câu, và so sánh LEAD vs REFRESH.
- Dán **Gold summary** để app chấm **ROUGE-1/2/L** (rougeLsum + stemmer, đúng metric của paper).

## Cấu trúc

```
app/
  streamlit_app.py     # UI
  refresh_model.py     # kiến trúc REFRESH + tokenizer + inference (khớp notebook)
  requirements.txt
  model/               # <- đặt refresh_best.pt + vocab.json + meta.json vào đây
  sample_texts/        # bài mẫu .txt (sample/ + gold/) - copy-paste thủ công
```

## Ghi chú faithful theo paper

- `refresh_model.py` copy **chính xác** kiến trúc trong notebook (CNN 1–7 ×50 -> 350; doc LSTM 600 đảo chiều; extractor LSTM init từ doc state -> Linear 2) nên `refresh_best.pt` load khớp.
- App **không cần GloVe**: embedding đã train nằm trong checkpoint; chỉ cần `vocab.json` để map từ -> id.
- ROUGE trong app dùng `rougeLsum` + Porter stemmer = khớp pyrouge `-m` của paper.
