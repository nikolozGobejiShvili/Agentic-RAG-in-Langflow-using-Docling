# Agentic RAG System (Docling → Chroma → Groq)

Fully Windows-compatible | PDF Ingestion | Structured Chunking | Vector Search | Source-grounded Answers

This project implements a clean and reliable Retrieval-Augmented Generation (RAG) pipeline using:

Docling for high-quality PDF parsing

TF-IDF + SVD embeddings (100% CPU / Windows friendly)

ChromaDB as a persistent vector store

Groq LLMs for fast, low-latency responses

It is designed to run on Windows without GPU acceleration, ONNXRuntime, or Torch.

# Features

Robust PDF → Markdown extraction
Uses Docling first, and automatically falls back to PyMuPDF for scanned or problematic PDFs.

Intelligent text chunking
Header-aware splitting preserves semantic structure and improves retrieval accuracy.

Persistent vector database (Chroma)
Saves your embeddings in ./vectorstore so ingestion is not repeated.

CPU-friendly embeddings
TF-IDF + TruncatedSVD works everywhere with no DLL issues.

Interactive RAG chat agent
Answers grounded in real document chunks with source citations.

One-click setup
A Windows setup script (setup_win.ps1) installs everything, ingests PDFs, and launches the agent.

# Project Structure
data/                 # Input PDFs
vectorstore/          # Persisted embeddings
scripts/
   ingest.py          # PDF → chunks → vectorstore
   run_agent.py       # Interactive RAG chat
   setup_win.ps1      # Full installation + run script
utils.py              # Embeddings + Chroma helpers
requirements.txt
README.md

# Quick Start (Windows)
1. Add PDFs

Place your documents inside the data/ directory.

2. Add API keys

Create a .env file:

GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

3. Run the system

From the project root:

powershell -ExecutionPolicy Bypass -File .\scripts\setup_win.ps1


This will:

Install all dependencies

Remove any old vectorstore

Ingest your PDFs

Launch the chat agent

# How the System Works
1. PDF Parsing

Docling 2.62.0 is used for structure-preserving extraction

Automatic fallback to PyMuPDF ensures all PDFs are processed

2. Chunking

Chunks use:

Size: 1000 tokens

Overlap: 150

Header-aware separators

Rich metadata (source, title, chunk_id)

3. Embeddings (CPU-safe)

Instead of heavy DL embeddings (Torch/ONNX), we use:

TF-IDF Vectorizer

TruncatedSVD (LSA)

Fast, stable, zero-dependency issues.

4. Vector Database (ChromaDB)

Chroma stores and retrieves chunks via MMR-based semantic search.

5. Query Engine (Groq + LangChain)

The agent uses a RetrievalQA chain with:

Groq LLM (llama-3.1-8b-instant)

Clean answer formatting

Inline source citations

# Example Questions to Ask

You can ask questions such as:

"Explain multi-head attention from the Transformer paper."

"Show a minimal Django REST Framework Book API."

"Compare bagging and boosting."

"What is the 2021 WHO PM2.5 annual limit?"

"Summarize chapter 3 of Hands-On Machine Learning."

# Design Decisions (For Academic Submission)

Docling chosen for high-fidelity document parsing

TF-IDF/SVD chosen due to Windows compatibility and no GPU dependency

Header-aware chunking improves semantic retrieval accuracy

Chroma selected for lightweight, persistent local vector storage

Groq LLM ensures fast inference with high accuracy

Separation of stages (ingest vs. agent) improves clarity and testing

Metadata-rich chunks ensure source-grounded citation

# Troubleshooting
Issue	Meaning	Fix
model_decommissioned	Groq model deprecated	Switch to llama-3.1-8b-instant
DLL load failed	ONNXRuntime/Torch issue	Not used in this build — you are safe
Chroma warnings	Deprecation messages	Safe to ignore
.

# If you want, I can also generate:

 A full project report (for school submission)
 A PowerPoint presentation
 A demo video script
 Extra test questions and answers

Just tell me!
