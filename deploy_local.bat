@echo off
echo ==============================================
echo 🚀 Deploying All AI Projects Locally...
echo ==============================================

cd /d "%~dp0"

echo [1/3] Starting PrivateDocs RAG (Port 8501)...
start "PrivateDocs RAG" cmd /c "cd privatedocs-rag && venv\Scripts\python.exe -m streamlit run app.py --server.port 8501"

echo [2/3] Starting CineAI Movie Recommender (Port 8502)...
start "Movie Recommender" cmd /c "cd movie-recommender && venv\Scripts\python.exe -m streamlit run app.py --server.port 8502"

echo [3/3] Starting LightRAG Graph Server (Port 9621)...
start "LightRAG" cmd /c "cd lightrag-app\LightRAG && .venv\Scripts\lightrag-server.exe"

echo.
echo ✅ All services launched in the background!
echo Opening Dashboard...
start dashboard.html

echo Done! You can close this window.
timeout /t 5 >nul
