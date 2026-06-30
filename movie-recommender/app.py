"""
Movie Recommendation System
Collaborative Filtering (SVD Matrix Factorization) on MovieLens 100K
+ Groq LLM for natural language movie Q&A
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR      = Path("ml-100k")
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── Genre tag colors ──────────────────────────────────────────────────────────
GENRE_COLORS = {
    "Action":    "#ef4444", "Adventure": "#f97316", "Animation": "#eab308",
    "Children's":"#84cc16", "Comedy":    "#22c55e", "Crime":     "#14b8a6",
    "Documentary":"#06b6d4","Drama":     "#3b82f6", "Fantasy":   "#8b5cf6",
    "Film-Noir": "#a855f7", "Horror":    "#ec4899", "Musical":   "#f43f5e",
    "Mystery":   "#6366f1", "Romance":   "#e11d48", "Sci-Fi":    "#0ea5e9",
    "Thriller":  "#64748b", "War":       "#78716c", "Western":   "#d97706",
    "unknown":   "#374151",
}
GENRE_COLS = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
    "Romance","Sci-Fi","Thriller","War","Western",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0a2e 40%, #16213e 100%);
    min-height: 100vh;
}
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem;
}
.hero-sub { color: rgba(255,255,255,0.45); font-size: 1rem; margin-bottom: 2rem; }

.movie-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 0.7rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #f59e0b, #ef4444);
}
.movie-rank {
    font-size: 2rem; font-weight: 800;
    color: rgba(255,255,255,0.15);
    position: absolute; top: 0.8rem; right: 1rem;
}
.movie-title { font-size: 1.05rem; font-weight: 700; color: white; margin-bottom: 0.3rem; }
.movie-year  { color: rgba(255,255,255,0.4); font-size: 0.82rem; }
.star-bar    { color: #f59e0b; font-size: 1rem; letter-spacing: 0.05em; }
.score-badge {
    background: linear-gradient(135deg, #f59e0b22, #ef444422);
    border: 1px solid rgba(245,158,11,0.3);
    color: #f59e0b; font-weight: 700; font-size: 0.85rem;
    padding: 0.2rem 0.7rem; border-radius: 999px; display: inline-block;
}
.genre-tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.1rem;
    color: white;
}
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #ef4444) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
.chat-user {
    background: linear-gradient(135deg, #f59e0b22, #ef444422);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1.1rem; margin: 0.5rem 0;
    max-width: 80%; margin-left: auto;
    color: rgba(255,255,255,0.9); font-size: 0.93rem;
}
.chat-ai {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1.1rem; margin: 0.5rem 0;
    max-width: 85%; color: rgba(255,255,255,0.88); font-size: 0.93rem;
    line-height: 1.6;
}
label, .stMarkdown, p, h1, h2, h3, li { color: rgba(255,255,255,0.85) !important; }
h1, h2, h3 { color: white !important; }
.stSelectbox > div > div, .stSlider, .stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important; border-radius: 10px !important;
}
hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_movielens():
    if not DATA_DIR.exists():
        with st.spinner("📥 Downloading MovieLens 100K dataset..."):
            r = requests.get(MOVIELENS_URL, timeout=60)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(".")
    return True


@st.cache_data(show_spinner=False)
def load_data():
    download_movielens()
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|", header=None, encoding="latin-1",
        names=["movie_id","title","release_date","video_release","imdb_url"] + GENRE_COLS,
        usecols=["movie_id","title","release_date"] + GENRE_COLS,
    )
    movies["year"] = movies["release_date"].str.extract(r"(\d{4})").fillna("N/A")
    movies["genres"] = movies.apply(
        lambda r: [g for g in GENRE_COLS if r[g] == 1], axis=1
    )
    return ratings, movies


@st.cache_data(show_spinner=False)
def build_model(ratings):
    matrix = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)
    norm = matrix.subtract(matrix.mean(axis=1), axis=0)
    item_sim = cosine_similarity(norm.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
    return matrix, item_sim_df


def get_user_recommendations(user_id: int, matrix, item_sim_df, movies_df, n=10):
    if user_id not in matrix.index:
        return pd.DataFrame()
    user_row    = matrix.loc[user_id]
    rated       = user_row[user_row > 0].index.tolist()
    unrated     = user_row[user_row == 0].index.tolist()
    if not rated or not unrated:
        return pd.DataFrame()

    scores = {}
    for mid in unrated:
        if mid not in item_sim_df.columns:
            continue
        sims   = item_sim_df[mid][rated]
        rated_vals = user_row[rated]
        denom  = sims.abs().sum()
        if denom == 0:
            continue
        scores[mid] = (sims * rated_vals).sum() / denom

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    if not top:
        return pd.DataFrame()

    result = pd.DataFrame(top, columns=["movie_id", "score"])
    result = result.merge(movies_df[["movie_id","title","year","genres"]], on="movie_id")
    result["score"] = result["score"].round(3)
    return result.reset_index(drop=True)


def get_movie_info(movie_id: int, matrix, movies_df):
    row = movies_df[movies_df["movie_id"] == movie_id]
    if row.empty:
        return {}
    avg_rating = matrix[movie_id].mean() if movie_id in matrix.columns else 0
    count      = (matrix[movie_id] > 0).sum() if movie_id in matrix.columns else 0
    return {
        "title":  row.iloc[0]["title"],
        "year":   row.iloc[0]["year"],
        "genres": row.iloc[0]["genres"],
        "avg_rating": round(avg_rating, 2),
        "num_ratings": int(count),
    }


def stars(score, max_score=5):
    filled = round(score / max_score * 5)
    return "★" * filled + "☆" * (5 - filled)


# ── Session state ──────────────────────────────────────────────────────────────
if "chat"      not in st.session_state: st.session_state.chat      = []
if "recs"      not in st.session_state: st.session_state.recs      = None
if "sel_user"  not in st.session_state: st.session_state.sel_user  = 1

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading MovieLens dataset..."):
    ratings, movies = load_data()
    matrix, item_sim_df = build_model(ratings)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineAI Controls")
    st.markdown("---")

    api_key = st.text_input("🔑 Groq API Key", type="password",
                             value=os.getenv("GROQ_API_KEY",""),
                             placeholder="gsk_...")

    st.markdown("---")
    st.markdown("### 👤 User Selection")
    user_id = st.number_input(
        "User ID", min_value=1, max_value=int(matrix.index.max()),
        value=st.session_state.sel_user, step=1,
    )
    st.session_state.sel_user = user_id

    n_recs = st.slider("# Recommendations", 5, 20, 10)

    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Computing recommendations..."):
            st.session_state.recs = get_user_recommendations(
                user_id, matrix, item_sim_df, movies, n_recs
            )

    st.markdown("---")
    # User stats
    user_ratings = ratings[ratings["user_id"] == user_id]
    st.markdown("### 📊 User Stats")
    st.metric("Movies Rated", len(user_ratings))
    if len(user_ratings):
        st.metric("Avg Rating Given", f"{user_ratings['rating'].mean():.2f} ⭐")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);text-align:center">
    MovieLens 100K · 100,000 ratings<br>943 users · 1,682 movies
    </div>""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎬 CineAI Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Collaborative Filtering · 100K ratings · Powered by Groq AI</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "💬 AI Movie Chat", "📊 Dataset Insights"])

# ── Tab 1: Recommendations ────────────────────────────────────────────────────
with tab1:
    if st.session_state.recs is None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="glass-card"><h3>🧮 Algorithm</h3><p style="color:rgba(255,255,255,0.5);font-size:0.9rem">Item-based Collaborative Filtering using cosine similarity on user-item rating matrix.</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card"><h3>📦 Dataset</h3><p style="color:rgba(255,255,255,0.5);font-size:0.9rem">MovieLens 100K — 100,000 ratings from 943 users on 1,682 movies.</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="glass-card"><h3>⚡ How to use</h3><p style="color:rgba(255,255,255,0.5);font-size:0.9rem">Select a User ID in the sidebar and click "Get Recommendations".</p></div>', unsafe_allow_html=True)
    else:
        recs = st.session_state.recs
        if recs.empty:
            st.warning("No recommendations found for this user. Try a different User ID.")
        else:
            st.markdown(f"### 🎯 Top {len(recs)} Picks for User #{user_id}")
            for i, row in recs.iterrows():
                genres_html = "".join(
                    f'<span class="genre-tag" style="background:{GENRE_COLORS.get(g, "#374151")}88;border:1px solid {GENRE_COLORS.get(g,"#374151")};">{g}</span>'
                    for g in row["genres"][:4]
                )
                score_pct = min(row["score"] / 5 * 100, 100)
                st.markdown(f"""
                <div class="movie-card">
                    <span class="movie-rank">#{i+1}</span>
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-year">📅 {row['year']}</div>
                    <div style="margin:0.4rem 0">{genres_html}</div>
                    <div style="display:flex;align-items:center;gap:1rem;margin-top:0.5rem">
                        <span class="score-badge">Score: {row['score']:.3f}</span>
                        <div style="flex:1;background:rgba(255,255,255,0.08);border-radius:999px;height:6px;">
                            <div style="width:{score_pct:.0f}%;background:linear-gradient(90deg,#f59e0b,#ef4444);height:6px;border-radius:999px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Tab 2: AI Movie Chat ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### 💬 Ask the Movie AI")
    st.caption("Ask about movies, genres, recommendations, or anything cinema-related.")

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🎬 {msg["content"]}</div>', unsafe_allow_html=True)

    suggestions_chat = [
        "What movies should I watch tonight?",
        "Top 5 sci-fi movies of the 90s?",
        "Recommend a good thriller",
        "Best animated films for adults?",
    ]
    scols = st.columns(len(suggestions_chat))
    clicked = None
    for col, sug in zip(scols, suggestions_chat):
        with col:
            if st.button(sug, key=f"csug_{sug[:10]}", use_container_width=True):
                clicked = sug

    with st.form("movie_chat_form", clear_on_submit=True):
        col_q, col_s = st.columns([5,1])
        with col_q:
            q = st.text_input("Ask about movies...", label_visibility="collapsed")
        with col_s:
            go = st.form_submit_button("Ask ➤", use_container_width=True)

    final_q = (q if go and q.strip() else None) or clicked

    if final_q:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat.append({"role": "user", "content": final_q})

            # Build context from dataset
            top_movies = movies.nlargest(20, "movie_id")[["title","year","genres"]].to_string(index=False)
            recs_context = ""
            if st.session_state.recs is not None and not st.session_state.recs.empty:
                recs_context = f"\nUser #{user_id}'s top recommendations: " + \
                    ", ".join(st.session_state.recs["title"].tolist()[:5])

            system_msg = f"""You are CineAI, a passionate and knowledgeable movie recommendation assistant.
You have access to the MovieLens 100K dataset with 1,682 movies and 943 users.
{recs_context}
Be enthusiastic, concise, and give great movie suggestions. Use emojis sparingly."""

            with st.spinner("🎬 Thinking..."):
                try:
                    client = Groq(api_key=api_key)
                    messages = [{"role": "system", "content": system_msg}]
                    for m in st.session_state.chat[-6:]:
                        messages.append({"role": m["role"], "content": m["content"]})

                    resp = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.7,
                    )
                    answer = resp.choices[0].message.content
                    st.session_state.chat.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.chat.append({"role": "assistant", "content": f"❌ {e}"})
            st.rerun()

# ── Tab 3: Dataset Insights ───────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", f"{len(ratings):,}")
    col2.metric("Unique Users",  f"{ratings['user_id'].nunique():,}")
    col3.metric("Unique Movies", f"{ratings['movie_id'].nunique():,}")
    col4.metric("Avg Rating",    f"{ratings['rating'].mean():.2f} ⭐")

    st.markdown("---")
    st.markdown("#### 🎭 Most-Rated Movies")
    top_rated = (
        ratings.groupby("movie_id")
        .agg(count=("rating","count"), avg=("rating","mean"))
        .reset_index()
        .merge(movies[["movie_id","title"]], on="movie_id")
        .nlargest(10, "count")[["title","count","avg"]]
        .rename(columns={"title":"Movie","count":"# Ratings","avg":"Avg ⭐"})
    )
    top_rated["Avg ⭐"] = top_rated["Avg ⭐"].round(2)
    st.dataframe(top_rated, use_container_width=True, hide_index=True)

    st.markdown("#### 🎬 Genre Distribution")
    genre_counts = {g: movies[g].sum() for g in GENRE_COLS if movies[g].sum() > 0}
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=["Genre","Count"]).sort_values("Count", ascending=False)
    st.bar_chart(genre_df.set_index("Genre"))
