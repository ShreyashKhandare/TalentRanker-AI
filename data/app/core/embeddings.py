"""Semantic embedding helpers using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_APP_MODEL: Any | None = None


def set_model(model: Any) -> None:
    """Inject a preloaded sentence-transformer model (e.g. from FastAPI app.state)."""
    global _APP_MODEL
    _APP_MODEL = model


@lru_cache(maxsize=1)
def _get_model() -> Any:
    """Load the embedding model once and cache it."""
    return sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Return vector embedding for input text."""
    if not isinstance(text, str) or not text.strip():
        return []

    model = _APP_MODEL if _APP_MODEL is not None else _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def compute_similarity(text1: str, text2: str) -> float:
    """Return cosine similarity between two texts."""
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        from sentence_transformers import util

        model = _APP_MODEL if _APP_MODEL is not None else _get_model()
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())
    except Exception:
        return 0.0


if __name__ == "__main__":
    a = "Machine Learning"
    b = "Artificial Intelligence"
    score = compute_similarity(a, b)
    print(f"Similarity('{a}', '{b}') = {score:.4f}")
