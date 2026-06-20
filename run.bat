@echo off
REM Chay app Streamlit bang Python trong VENV (khong dung Python tong).
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo [!] Chua co venv. Tao va cai dat:
  echo     python -m venv .venv
  echo     .venv\Scripts\python -m pip install -r requirements.txt
  echo     .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  pause
  exit /b 1
)
REM Tai san NLTK punkt (1 lan, no-op neu da co) -> demo khong bi khung lai lan dau
".venv\Scripts\python.exe" -c "import nltk;[nltk.download(p,quiet=True) for p in ('punkt','punkt_tab')]"
".venv\Scripts\streamlit.exe" run streamlit_app.py
