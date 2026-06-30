"""
PrivateDocs RAG — Streamlit App
Beautiful UI to upload docs, chat with them using Groq LLM + ChromaDB
"""

import os
import tempfile
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODELS  = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrivateDocs RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Main container */
.main .block-container { padding-top: 2rem; }

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Title */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Chat messages */
.user-bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.6rem 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
    font-size: 0.95rem;
}
.ai-bubble {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.92);
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.6rem 0;
    max-width: 85%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    font-size: 0.95rem;
    line-height: 1.6;
}
.ai-label {
    font-size: 0.72rem;
    color: #a78bfa;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}
.user-label {
    font-size: 0.72rem;
    color: #60a5fa;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    text-align: right;
}

/* Source cards */
.source-card {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
}

/* Status badges */
.badge-green {
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    color: #34d399;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
}
.badge-yellow {
    background: rgba(251,191,36,0.15);
    border: 1px solid rgba(251,191,36,0.4);
    color: #fbbf24;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* Input box */
.stTextInput > div > div > input, .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 12px !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}

/* File uploader */
.stFileUploader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Text colors */
label, .stMarkdown, p, h1, h2, h3, li {
    color: rgba(255,255,255,0.85) !important;
}
h1, h2, h3 { color: white !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 3px; }

/* Spinner */
.stSpinner > div { border-color: #a78bfa !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "vectorstore"    not in st.session_state: st.session_state.vectorstore    = None
if "chain"          not in st.session_state: st.session_state.chain          = None
if "doc_count"      not in st.session_state: st.session_state.doc_count      = 0
if "chunk_count"    not in st.session_state: st.session_state.chunk_count    = 0
if "memory"         not in st.session_state: st.session_state.memory         = None


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_chunk(uploaded_files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    with tempfile.TemporaryDirectory() as tmp:
        for uf in uploaded_files:
            fpath = Path(tmp) / uf.name
            fpath.write_bytes(uf.read())
            try:
                if uf.name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(str(fpath))
                elif uf.name.lower().endswith((".txt", ".md")):
                    loader = TextLoader(str(fpath), encoding="utf-8")
                else:
                    continue
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                st.warning(f"Could not parse {uf.name}: {e}")
    return all_chunks


def build_chain(api_key: str, model_name: str, vectorstore):
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        max_tokens=2048,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=6,
    )
    st.session_state.memory = memory

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # API Key
    api_key_input = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        help="Free at console.groq.com",
    )

    # Model selector
    model_choice = st.selectbox("🤖 Model", GROQ_MODELS, index=0)

    st.markdown("---")
    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop PDFs or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("🚀 Build Index", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear All", use_container_width=True)

    if build_btn:
        if not api_key_input:
            st.error("Please enter your Groq API key.")
        elif not uploaded_files:
            st.error("Upload at least one document.")
        else:
            with st.spinner("📥 Ingesting documents..."):
                chunks = load_and_chunk(uploaded_files)
                if chunks:
                    embeddings = get_embeddings()
                    if os.path.exists(CHROMA_PATH):
                        shutil.rmtree(CHROMA_PATH)
                    vs = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=CHROMA_PATH,
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.doc_count   = len(uploaded_files)
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.chat_history = []
                    st.session_state.chain = build_chain(api_key_input, model_choice, vs)
                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")
                else:
                    st.error("No text could be extracted from the files.")

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.vectorstore  = None
        st.session_state.chain        = None
        st.session_state.doc_count    = 0
        st.session_state.chunk_count  = 0
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        st.rerun()

    st.markdown("---")

    # Status
    st.markdown("### 📊 Status")
    ready = st.session_state.chain is not None

    if ready:
        st.markdown('<span class="badge-green">● READY</span>', unsafe_allow_html=True)
        st.metric("Documents", st.session_state.doc_count)
        st.metric("Chunks", st.session_state.chunk_count)
        st.metric("Messages", len(st.session_state.chat_history))
    else:
        st.markdown('<span class="badge-yellow">○ No Index</span>', unsafe_allow_html=True)
        st.caption("Upload documents and click Build Index to start.")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); text-align:center;">
    🔒 100% Private · Data stays local<br>
    ⚡ Powered by Groq LPU™ Inference
    </div>
    """, unsafe_allow_html=True)


# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🧠 PrivateDocs RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Chat with your documents · Private · Fast · Powered by Groq</p>', unsafe_allow_html=True)

# ── Welcome screen ─────────────────────────────────────────────────────────────
if not st.session_state.chain:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📄 Upload</h3>
            <p style="color:rgba(255,255,255,0.6);font-size:0.9rem;">
            Drop in PDFs, notes, textbooks, research papers, or any text file.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🔍 Index</h3>
            <p style="color:rgba(255,255,255,0.6);font-size:0.9rem;">
            Documents are chunked, embedded locally, and stored in ChromaDB — no cloud.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>💬 Chat</h3>
            <p style="color:rgba(255,255,255,0.6);font-size:0.9rem;">
            Ask anything. Groq's blazing-fast LLM answers using only your documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="text-align:center; margin-top:2rem;">
        <h2 style="margin:0 0 0.5rem 0;">👈 Get Started</h2>
        <p style="color:rgba(255,255,255,0.5);margin:0;">
            Enter your Groq API key, upload documents, and click <strong>Build Index</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat interface ──────────────────────────────────────────────────────────
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:2rem;">
                <h3 style="margin:0 0 0.5rem 0;">✅ Index Ready!</h3>
                <p style="color:rgba(255,255,255,0.5);margin:0;">
                    Ask me anything about your documents below.
                </p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-label">You</div><div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-label">🧠 AI · {model_choice}</div><div class="ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        for src in msg["sources"]:
                            page_label = f" · Page {src.metadata.get('page', '?') + 1}" if 'page' in src.metadata else ""
                            source_name = Path(src.metadata.get("source", "Unknown")).name
                            st.markdown(f'<div class="source-card">📄 <strong>{source_name}</strong>{page_label}<br><span style="font-size:0.8rem;opacity:0.7">{src.page_content[:200]}…</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input row ───────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col_q, col_send = st.columns([6, 1])
        with col_q:
            question = st.text_input(
                "Ask a question",
                placeholder="What does this document say about...?",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    # ── Quick suggestions ───────────────────────────────────────────────────────
    suggestions = [
        "Summarize the key points",
        "What are the main conclusions?",
        "List all important dates or numbers",
        "Explain in simple terms",
    ]
    st.markdown("**💡 Quick Questions:**")
    scols = st.columns(len(suggestions))
    clicked_suggestion = None
    for i, (col, sug) in enumerate(zip(scols, suggestions)):
        with col:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                clicked_suggestion = sug

    # ── Handle send ─────────────────────────────────────────────────────────────
    final_q = question if submitted and question.strip() else clicked_suggestion

    if final_q:
        st.session_state.chat_history.append({"role": "user", "content": final_q})

        with st.spinner("🔍 Searching your documents..."):
            try:
                result = st.session_state.chain.invoke({"question": final_q})
                answer  = result.get("answer", "I couldn't find an answer.")
                sources = result.get("source_documents", [])

                # Deduplicate sources
                seen, unique_sources = set(), []
                for doc in sources:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(doc)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": unique_sources,
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                    "sources": [],
                })

        st.rerun()
