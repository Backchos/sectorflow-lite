@echo off
echo 🚀 SectorFlow Lite 대시보드 시작 중...

REM 가상환경 활성화 (있는 경우)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Streamlit 실행 및 브라우저 자동 열기
python -m streamlit run examples/app_streamlit.py --server.headless false

pause


