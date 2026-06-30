@echo off
echo.
echo ========================================
echo    PrivateDocs RAG  -  Launching...
echo ========================================
echo.
cd /d "%~dp0"
call venv\Scripts\activate
echo [INFO] Starting Streamlit app...
echo [INFO] Opening at http://localhost:8501
echo.
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
