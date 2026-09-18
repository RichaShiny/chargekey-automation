from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)
        values = self._load().encode(
            list(texts), show_progress_bar=False, batch_size=64
        )
        values = np.asarray(values, dtype=float)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return values / norms
