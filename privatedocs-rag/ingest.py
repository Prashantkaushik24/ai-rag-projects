"""
PrivateDocs RAG — Ingest pipeline
Loads PDFs, chunks them, embeds with sentence-transformers, stores in ChromaDB
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
DOCS_PATH   = "documents"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(source_dir: str):
    docs = []
    path = Path(source_dir)
    if not path.exists():
        path.mkdir(parents=True)
        return docs

    for fpath in path.glob("**/*"):
        if fpath.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(fpath))
                docs.extend(loader.load())
            except Exception as e:
                print(f"[WARN] Could not load {fpath}: {e}")
        elif fpath.suffix.lower() in (".txt", ".md"):
            try:
                loader = TextLoader(str(fpath), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"[WARN] Could not load {fpath}: {e}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest(source_dir: str = DOCS_PATH, reset: bool = False):
    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("[INFO] Cleared existing vector store.")

    print(f"[INFO] Loading documents from '{source_dir}' ...")
    docs = load_documents(source_dir)
    if not docs:
        print("[WARN] No documents found. Add PDF/TXT files to the 'documents/' folder.")
        return 0

    print(f"[INFO] Loaded {len(docs)} pages. Splitting ...")
    chunks = split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks. Embedding ...")

    embeddings = get_embeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"[OK] Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'.")
    return len(chunks)


if __name__ == "__main__":
    ingest(reset=True)
