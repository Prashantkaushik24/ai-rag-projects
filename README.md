# 🤖 RAG & Recommendation AI Projects

> Powered by **Groq LPU™** · 100% Private · Runs on your ASUS RTX 3050

---

## 📁 Project Structure

```
rag-projects/
├── privatedocs-rag/       # 🔵 Chat with your own PDFs
│   ├── app.py             # Main Streamlit app
│   ├── ingest.py          # Document ingestion pipeline
│   ├── requirements.txt
│   ├── .env               # ← Put your Groq API key here
│   ├── launch.bat         # ← Double-click to start!
│   └── venv/
│
├── movie-recommender/     # 🟡 Movie Recommendation System
│   ├── app.py             # Main Streamlit app
│   ├── requirements.txt
│   ├── launch.bat         # ← Double-click to start!
│   └── venv/
│
└── lightrag-app/          # 🟣 Knowledge Graph RAG
    └── LightRAG/          # Cloned from GitHub
```

---

## 🚀 Quick Start

### Step 1 — Add your Groq API Key

Edit `privatedocs-rag/.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 2 — Launch Projects

| Project | How to run |
|---------|-----------|
| 🔵 PrivateDocs RAG | Double-click `privatedocs-rag/launch.bat` |
| 🟡 Movie Recommender | Double-click `movie-recommender/launch.bat` |

---

## 🔵 PrivateDocs RAG

**Chat with any PDF, TXT, or Markdown file using a local AI.**

Features:
- 📄 Upload multiple files at once
- 🔍 Semantic search with `sentence-transformers/all-MiniLM-L6-v2`
- 🗄️ ChromaDB local vector store (no cloud)
- 💬 Conversational memory (remembers last 6 exchanges)
- 📎 Source citations with page numbers
- ⚡ Groq LLM (`llama-3.3-70b-versatile` by default)

Available models on Groq:
- `llama-3.3-70b-versatile` — Best quality
- `llama3-8b-8192` — Faster
- `mixtral-8x7b-32768` — Great for long documents
- `gemma2-9b-it` — Google's model

---

## 🟡 Movie Recommender (CineAI)

**Collaborative Filtering on MovieLens 100K dataset.**

Features:
- 🧮 Item-based Collaborative Filtering (Cosine Similarity)
- 📊 100,000 ratings · 943 users · 1,682 movies
- 🎯 Personalized top-N recommendations per user
- 💬 Groq AI movie chatbot
- 📊 Dataset insights & genre distribution

---

## 🟣 LightRAG (Knowledge Graph RAG)

```bash
cd lightrag-app/LightRAG
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[api]"

# Set your Groq key:
set GROQ_API_KEY=gsk_your_key_here

# Run server
lightrag-server
```
Then open: http://localhost:9621

---

## 💡 Tips

- **Free Groq tier**: ~14,400 requests/day on llama-3.3-70b
- **Best for RAG**: Upload 5–20 page PDFs for best results
- **Context window**: Groq models support up to 32k tokens
