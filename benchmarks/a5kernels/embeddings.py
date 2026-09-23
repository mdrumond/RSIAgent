"""Pinned, offline-capable production embeddings for the A5 knowledge DB."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .knowledge import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION

EXPECTED_EMBEDDING_DIMENSION = 384


class EmbeddingLoadError(RuntimeError):
    """Raised when the pinned model cannot be loaded with verified provenance."""


class PinnedBGEEmbeddings:
    """CPU BGE embeddings loaded at one immutable Hugging Face revision.

    Loading is local-cache-only by default. Callers must opt into downloads
    explicitly, making an experiment's network behavior visible at setup time.
    The heavyweight dependencies are imported lazily so lexical-only tooling and
    custom ``EmbeddingBackend`` implementations remain lightweight.
    """

    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION
    dimension = EXPECTED_EMBEDDING_DIMENSION

    def __init__(
        self,
        *,
        batch_size: int = 32,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._torch = torch_module
        self._tokenizer: Any | None = None
        self._encoder: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None:
            return self._tokenizer, self._encoder, self._torch
        try:
            if self._torch is None:
                import torch

                self._torch = torch
            if self._tokenizer_loader is None or self._model_loader is None:
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer_loader = AutoTokenizer.from_pretrained
                self._model_loader = AutoModel.from_pretrained
            kwargs: dict[str, Any] = {
                "revision": self.revision,
                "local_files_only": self.local_files_only,
                "trust_remote_code": False,
            }
            if self.cache_dir is not None:
                kwargs["cache_dir"] = str(self.cache_dir.resolve())
            tokenizer = self._tokenizer_loader(self.model, **kwargs)
            encoder = self._model_loader(self.model, **kwargs)
        except Exception as exc:
            mode = "local cache" if self.local_files_only else "Hugging Face"
            raise EmbeddingLoadError(
                f"could not load {self.model}@{self.revision} from {mode}"
            ) from exc

        config = getattr(encoder, "config", None)
        commit = getattr(config, "_commit_hash", None)
        hidden_size = getattr(config, "hidden_size", None)
        if commit != self.revision:
            raise EmbeddingLoadError(
                f"model provenance is unverified: expected revision {self.revision}, got {commit!r}"
            )
        if hidden_size != self.dimension:
            raise EmbeddingLoadError(
                f"unexpected embedding dimension: expected {self.dimension}, got {hidden_size!r}"
            )
        try:
            encoder.to("cpu")
            encoder.eval()
            self._torch.use_deterministic_algorithms(True)
        except Exception as exc:
            raise EmbeddingLoadError("could not configure deterministic CPU inference") from exc
        self._tokenizer, self._encoder = tokenizer, encoder
        return tokenizer, encoder, self._torch

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all embedding inputs must be strings")
        tokenizer, encoder, torch = self._load()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            try:
                tokens = {key: value.to("cpu") for key, value in tokens.items()}
                with torch.inference_mode():
                    output = encoder(**tokens)
                states = output.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(states.dtype)
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                rows = pooled.detach().cpu().tolist()
            except Exception as exc:
                raise EmbeddingLoadError("BGE inference returned an invalid output") from exc
            for row in rows:
                if len(row) != self.dimension or not all(math.isfinite(value) for value in row):
                    raise EmbeddingLoadError("BGE inference returned an invalid embedding")
                norm = math.sqrt(sum(value * value for value in row))
                if not norm:
                    raise EmbeddingLoadError("BGE inference returned a zero embedding")
                embeddings.append([value / norm for value in row])
        if len(embeddings) != len(texts):
            raise EmbeddingLoadError("BGE inference returned the wrong number of embeddings")
        return embeddings
