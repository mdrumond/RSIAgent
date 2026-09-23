"""A5 kernel exploration and benchmark support."""

from benchmarks.a5kernels.bz import BZSessionAdapter, ProfileCommandExecutor
from benchmarks.a5kernels.fixtures import Language, fixture_for
from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    CollectionManifest,
    EmbeddingBackend,
    KnowledgeDB,
    SearchHit,
)
from benchmarks.a5kernels.protocol import RunRequest, VerifiedResult
from benchmarks.a5kernels.runner import A5KernelRunner, ExecutionBackend

__all__ = [
    "A5KernelRunner",
    "BZSessionAdapter",
    "CollectionManifest",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "EmbeddingBackend",
    "ExecutionBackend",
    "KnowledgeDB",
    "Language",
    "ProfileCommandExecutor",
    "RunRequest",
    "SearchHit",
    "VerifiedResult",
    "fixture_for",
]
