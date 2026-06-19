Đặt 3 file checkpoint REFRESH vào ĐÚNG thư mục này (app/model/):

    refresh_best.pt     <- state_dict của REFRESHModel
    vocab.json          <- danh sách vocab (index = id)
    meta.json           <- cấu hình model + tiền xử lý

Cách lấy 3 file này:
  1. Mở notebook REFRESH_train_for_app.ipynb trên Google Colab.
  2. Runtime -> Run all  (train ~45-90 phút).
  3. Cell cuối "EXPORT FOR APP" tạo & tự tải về  refresh_app_bundle.zip.
  4. Giải nén zip, copy 3 file vào thư mục app/model/ này.
  5. Chạy lại app (.venv\Scripts\streamlit run streamlit_app.py) -> sidebar hiện "Đã nạp mô hình REFRESH".

Khi chưa có checkpoint, app vẫn chạy được baseline LEAD để test giao diện.