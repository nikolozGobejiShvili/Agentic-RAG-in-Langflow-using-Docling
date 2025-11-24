# Agentic RAG System (Docling → Chroma → Groq → Langflow)

Fully Windows-compatible · PDF Ingestion · Structured Chunking · Vector Search · Source-grounded Answers · Langflow UI

This project implements a clean and reliable Retrieval-Augmented Generation (RAG) pipeline using:

- **Docling** for high-quality PDF parsing  
- **TF-IDF + SVD** embeddings (100% CPU / Windows friendly)  
- **ChromaDB** as a persistent vector store  
- **Groq LLMs** for fast, low-latency responses  
- **Langflow** as a visual UI to chat with your documents and inspect the RAG graph

It is designed to run on Windows without GPU acceleration, ONNXRuntime, or Torch.

---

## Features

### Core RAG Pipeline

- **Robust PDF → Markdown extraction**  
  Uses **Docling** first, and automatically falls back to **PyMuPDF** for scanned or problematic PDFs.

- **Intelligent text chunking**  
  Header-aware splitting preserves semantic structure and improves retrieval accuracy.

- **Persistent vector database (Chroma)**  
  Saves your embeddings in `./vectorstore` so ingestion is not repeated.

- **CPU-friendly embeddings**  
  TF-IDF + TruncatedSVD works everywhere with no DLL issues.

- **Interactive RAG chat agent (CLI)**  
  Answers grounded in real document chunks with source citations.

### Langflow Integration

- **Langflow UI RAG agent**  
  Visual flow that uses **existing Chroma vectorstore** (no re-ingestion inside Langflow).

- **Chat-first interface**  
  Flow built around:
  ```text
  Chat Input → Text Input → Chroma DB → Parser (Stringify) → Prompt Template → Groq → Type Convert → Chat Output
data/                 # Input PDFs
vectorstore/          # Persisted Chroma embeddings (created by ingest.py)

ingest.py             # PDF → chunks → vectorstore (Docling + PyMuPDF + TF-IDF + SVD)
utils.py              # Embeddings + Chroma helpers (SklearnTfidfEmbeddings, get_chroma_store, etc.)
tfidf_svd.py          # Custom TF-IDF+SVD embeddings component for Langflow (optional)

run_langflow_client.py# (CLI) Client for calling a Langflow Flow via HTTP API

start_langflow.ps1    # Windows script to start Langflow with this project
scripts/

requirements.txt
README.md

