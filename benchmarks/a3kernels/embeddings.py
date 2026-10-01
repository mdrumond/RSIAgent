"""Pinned, offline embeddings for the A3 Ascend C knowledge corpus."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EXPECTED_EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_SNAPSHOT_SHA256 = (
    (
        "config.json",
        "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    ),
    (
        "model.safetensors",
        "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    ),
    (
        "special_tokens_map.json",
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    (
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "tokenizer_config.json",
        "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    ),
    (
        "vocab.txt",
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
)


class EmbeddingLoadError(RuntimeError):
    """The pinned embedding snapshot is unavailable or has invalid provenance."""


def validate_local_snapshot(cache_dir: Path) -> Path:
    """Validate the pinned Hugging Face snapshot without resolving the network."""
    snapshot = (
        cache_dir.resolve()
        / "models--BAAI--bge-small-en-v1.5"
        / "snapshots"
        / DEFAULT_EMBEDDING_REVISION
    )
    if not snapshot.is_dir():
        raise EmbeddingLoadError(
            f"pinned embedding snapshot {DEFAULT_EMBEDDING_REVISION} is not present"
        )
    missing = [
        name for name, _digest in _SNAPSHOT_SHA256
        if not (snapshot / name).is_file()
    ]
    if missing:
        raise EmbeddingLoadError(f"pinned embedding snapshot is incomplete: {', '.join(missing)}")
    for name, expected in _SNAPSHOT_SHA256:
        digest = hashlib.sha256()
        with (snapshot / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise EmbeddingLoadError(f"pinned snapshot content mismatch: {name}")
    return snapshot


class PinnedBGEEmbeddings:
    """Deterministic CPU BGE encoder pinned to one locally cached revision."""

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
        if not local_files_only:
            raise ValueError("A3 embeddings require local_files_only=True")
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._torch = torch_module
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        if self.cache_dir is None and (
            self._tokenizer_loader is None or self._model_loader is None
        ):
            raise EmbeddingLoadError(
                "cache_dir is required to authenticate the pinned embedding snapshot"
            )
        try:
            if self._torch is None:
                import torch

                self._torch = torch
            if self._tokenizer_loader is None:
                from transformers import AutoTokenizer

                self._tokenizer_loader = AutoTokenizer.from_pretrained
            if self._model_loader is None:
                from transformers import AutoModel

                self._model_loader = AutoModel.from_pretrained
            kwargs: dict[str, Any] = {
                "revision": self.revision,
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if self.cache_dir is not None:
                validate_local_snapshot(self.cache_dir)
                kwargs["cache_dir"] = str(self.cache_dir.resolve())
            tokenizer = self._tokenizer_loader(self.model, **kwargs)
            encoder = self._model_loader(self.model, **kwargs)
        except EmbeddingLoadError:
            raise
        except Exception as exc:
            raise EmbeddingLoadError(
                f"could not load {self.model}@{self.revision} from local cache"
            ) from exc
        config = getattr(encoder, "config", None)
        if getattr(config, "_commit_hash", None) != self.revision:
            raise EmbeddingLoadError("model provenance is unverified")
        if getattr(config, "hidden_size", None) != self.dimension:
            raise EmbeddingLoadError("unexpected embedding dimension")
        try:
            encoder.to("cpu").eval()
            self._torch.use_deterministic_algorithms(True)
        except Exception as exc:
            raise EmbeddingLoadError("could not configure deterministic CPU inference") from exc
        self._loaded = tokenizer, encoder, self._torch
        return self._loaded

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([QUERY_INSTRUCTION + text for text in texts])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    # Keep the conventional singular-query backend surface expected by callers.
    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("embedding query must be a string")
        return self.embed_queries([text])[0]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all embedding inputs must be strings")
        if not texts:
            return []
        tokenizer, encoder, torch = self._load()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            try:
                tokens = {name: value.to("cpu") for name, value in tokens.items()}
                with torch.inference_mode():
                    states = encoder(**tokens).last_hidden_state[:, 0]
                rows = states.detach().cpu().tolist()
            except Exception as exc:
                raise EmbeddingLoadError("BGE inference returned an invalid output") from exc
            for row in rows:
                if len(row) != self.dimension or not all(math.isfinite(value) for value in row):
                    raise EmbeddingLoadError("BGE inference returned an invalid embedding")
                norm = math.sqrt(sum(value * value for value in row))
                if not norm:
                    raise EmbeddingLoadError("BGE inference returned a zero embedding")
                vectors.append([value / norm for value in row])
        if len(vectors) != len(texts):
            raise EmbeddingLoadError("BGE inference returned the wrong number of embeddings")
        return vectors
