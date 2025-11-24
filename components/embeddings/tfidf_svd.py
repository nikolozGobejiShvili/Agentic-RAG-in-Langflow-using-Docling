from __future__ import annotations
import importlib.util
import os
import sys
from langflow.custom import Component
from langflow.io import Output, StrInput
from langchain_core.embeddings import Embeddings

class TfidfSvdEmbeddings(Component):
    display_name: str = "TF-IDF + SVD (local)"
    description: str = "Loads utils.SklearnTfidfEmbeddings (TF-IDF + TruncatedSVD) from your repo"
    icon: str = "SquareGanttChart"
    name: str = "TfidfSvdEmbeddings"

    inputs = [
        StrInput(
            name="model_path",
            display_name="Model Path",
            info="Path to persisted joblib (vectorizer + svd).",
            value="vectorstore/tfidf_svd.joblib",
            required=False,
        ),
        StrInput(
            name="project_root",
            display_name="Project Root (optional)",
            info="Absolute repo path if Langflow runs elsewhere (so we can import utils.py).",
            value="",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [Output(display_name="Embeddings", name="embeddings", method="build_embeddings")]

    def _import_utils(self, repo_root: str):
        # Try normal import first
        try:
            from utils import SklearnTfidfEmbeddings  # type: ignore
            return SklearnTfidfEmbeddings
        except Exception:
            pass
        # Fallbacks: <repo>/utils.py or <repo>/scripts/utils.py
        candidates = [
            os.path.join(repo_root, "utils.py"),
            os.path.join(repo_root, "scripts", "utils.py"),
        ]
        for path in candidates:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("utils_dyn", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["utils_dyn"] = module
                    spec.loader.exec_module(module)  # WHY: load even if not on sys.path
                    if hasattr(module, "SklearnTfidfEmbeddings"):
                        return getattr(module, "SklearnTfidfEmbeddings")
        raise RuntimeError(
            "SklearnTfidfEmbeddings not found. "
            "Place utils.py in repo root or scripts/, and set 'project_root' to the repo root."
        )

    def build_embeddings(self) -> Embeddings:  # WHY: Chroma needs a LangChain Embeddings
        root = (self.project_root or "").strip() or os.getcwd()
        if root not in sys.path:
            sys.path.append(root)
        SklearnTfidfEmbeddings = self._import_utils(root)
        model_path = self.model_path or "vectorstore/tfidf_svd.joblib"
        if not os.path.isabs(model_path):
            model_path = os.path.join(root, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TF-IDF model not found at {model_path}. Run ingestion first.")
        emb = SklearnTfidfEmbeddings(model_path=model_path, fit_corpus=None)  # type: ignore[call-arg]
        self.status = f"Ready (root={root}; model={os.path.basename(model_path)})"
        return emb