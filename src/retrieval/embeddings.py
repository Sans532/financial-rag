"""Dense embedding model wrapper (local sentence-transformers, no API cost)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import get_settings


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
