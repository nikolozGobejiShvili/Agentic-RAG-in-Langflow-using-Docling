"""
PDFs -> text (Docling fallback to PyMuPDF) -> chunk -> persist to Chroma.
Stores metadata: source, title, parser, chunk_id for inline citations.
"""
import os
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import get_chroma_store, get_embeddings

DATA_DIR = "data"


def _list_pdfs() -> List[str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]


def _read_docling(pdf_path: str) -> str:
    try:
        from docling.document_converter import DocumentConverter

        conv = DocumentConverter()
        res = conv.convert_single(pdf_path)
        return res.render_as_markdown()
    except Exception:
        return ""


def _read_pymupdf(pdf_path: str) -> str:
    try:
        import fitz

        txts: List[str] = []
        with fitz.open(pdf_path) as doc:
            for p in doc:
                txts.append(p.get_text("text"))
        return "\n\n".join(txts).strip()
    except Exception:
        return ""


def _read_pdf(pdf_path: str) -> Tuple[str, str]:
    txt = _read_docling(pdf_path)
    if txt and len(txt) > 50:
        return txt, "docling"
    txt = _read_pymupdf(pdf_path)
    return txt, "pymupdf"


def _make_documents(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        text, parser = _read_pdf(path)
        if not text or len(text) < 20:
            print(f" Skipping empty/image-only PDF: {os.path.basename(path)}")
            continue
        meta = {
            "source": os.path.abspath(path),
            "title": os.path.splitext(os.path.basename(path))[0],
            "parser": parser,
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def ingest_pdfs():
    pdfs = _list_pdfs()
    if not pdfs:
        print(" Put PDFs into ./data and run again.")
        return

    print(f" Found {len(pdfs)} PDFs.")
    base_docs = _make_documents(pdfs)
    if not base_docs:
        print(" Nothing parsed.")
        return
    print(f" Parsed {len(base_docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    chunks = splitter.split_documents(base_docs)
    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = i

    print(f"🔹 Split into {len(chunks)} chunks.")

    # Train TF-IDF on all chunk texts on first run, then persist vectorstore
    embeddings = get_embeddings(train_texts=[d.page_content for d in chunks])
    db = get_chroma_store(embedding_function=embeddings)
    db.add_documents(chunks)
    db.persist()
    print(" Vectorstore saved to ./vectorstore")


if __name__ == "__main__":
    ingest_pdfs()