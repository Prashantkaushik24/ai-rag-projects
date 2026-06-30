@echo off
echo.
echo ========================================
echo    CineAI Movie Recommender - Launching...
echo ========================================
echo.
cd /d "%~dp0"
call venv\Scripts\activate
echo [INFO] Starting Streamlit app...
echo [INFO] Opening at http://localhost:8502
echo.
streamlit run app.py --server.port 8502 --browser.gatherUsageStats false
pause
