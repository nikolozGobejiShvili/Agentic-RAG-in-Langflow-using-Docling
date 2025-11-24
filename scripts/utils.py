"""
Embeddings + Chroma VectorStore.
Primary: pure-CPU TF-IDF + TruncatedSVD (LSA) persisted to ./vectorstore/tfidf_svd.joblib

"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

VECTOR_DIR = "vectorstore"
TFIDF_MODEL_PATH = os.path.join(VECTOR_DIR, "tfidf_svd.joblib")

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD


def _ensure_vector_dir() -> str:
    os.makedirs(VECTOR_DIR, exist_ok=True)
    return VECTOR_DIR


@dataclass
class _SkBundle:
    vectorizer: "TfidfVectorizer"
    svd: "TruncatedSVD"


class SklearnTfidfEmbeddings(Embeddings):
    """
    TF-IDF -> TruncatedSVD to dense vectors (LSA).
    Persisted with joblib at TFIDF_MODEL_PATH.
    """

    def __init__(self, model_path: str, fit_corpus: Optional[List[str]] = None, n_components: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        import joblib

        self._joblib = joblib
        self._path = model_path

        if fit_corpus is not None:
            vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=float)
            X = vec.fit_transform(fit_corpus)
            k = min(n_components, max(2, X.shape[1] - 1))
            svd = TruncatedSVD(n_components=k, random_state=42)
            _ = svd.fit_transform(X)
            joblib.dump({"vectorizer": vec, "svd": svd}, model_path)
            self._vec, self._svd = vec, svd
        else:
            bundle: Dict[str, Any] = self._joblib.load(model_path)
            self._vec, self._svd = bundle["vectorizer"], bundle["svd"]

    def _embed(self, texts: Iterable[str]) -> List[List[float]]:
        import numpy as np

        X = self._vec.transform(list(texts))
        Xr = self._svd.transform(X)
        return Xr.astype(np.float32).tolist()

    # LangChain API
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def get_embeddings(train_texts: Optional[List[str]] = None) -> Embeddings:
    """
    If there's a persisted TF-IDF model (or train_texts provided) → SklearnTfidfEmbeddings.
    Otherwise raise; ingestion რაუნდში აუცილებლად ვაფიტებთ.
    """
    _ensure_vector_dir()
    if train_texts is not None or os.path.exists(TFIDF_MODEL_PATH):
        return SklearnTfidfEmbeddings(TFIDF_MODEL_PATH, fit_corpus=train_texts)
    raise RuntimeError(
        "No TF-IDF model found. Run ingestion first to train TF-IDF (creates vectorstore/tfidf_svd.joblib)."
    )


def get_chroma_store(embedding_function: Optional[Embeddings] = None) -> Chroma:
    if embedding_function is None:
        embedding_function = get_embeddings()
    return Chroma(persist_directory=_ensure_vector_dir(), embedding_function=embedding_function)