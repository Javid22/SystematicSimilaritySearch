from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "word2vec": MODEL_DIR / "systematic_similarity_word2vec.pkl",
    "fasttext": MODEL_DIR / "systematic_similarity_fasttext.pkl",
}


@dataclass
class LoadedModel:
    name: str
    embedding_model: Any
    document_vectors: np.ndarray
    documents: pd.DataFrame
    vector_size: int


def _extract_artifacts(package: Any) -> tuple[Any, Any, Any]:
    """Support either a packaged dict or a bare Gensim model (see test_word2vec.py)."""
    if isinstance(package, dict):
        model = (
            package.get("embedding_model")
            or package.get("word2vec_model")
            or package.get("fasttext_model")
            or package.get("model")
        )
        document_vectors = package.get("document_vectors")
        documents = package.get("documents")
    else:
        model, document_vectors, documents = package, None, None

    if model is None and hasattr(package, "wv"):
        model = package

    if model is None or not hasattr(model, "wv"):
        raise ValueError("Could not find a Gensim model in the pickle file.")

    return model, document_vectors, documents


def load_model(name: str) -> LoadedModel:
    if name not in MODEL_PATHS:
        raise ValueError(f"Unknown model '{name}'. Expected one of {list(MODEL_PATHS)}.")

    path = MODEL_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    package = joblib.load(path)
    model, document_vectors, documents = _extract_artifacts(package)

    if document_vectors is None or documents is None:
        raise ValueError(f"Pickle for '{name}' is missing document_vectors/documents.")

    return LoadedModel(
        name=name,
        embedding_model=model,
        document_vectors=np.asarray(document_vectors),
        documents=documents,
        vector_size=model.wv.vector_size,
    )
