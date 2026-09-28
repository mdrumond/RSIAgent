"""A5 kernel exploration and benchmark support."""

from benchmarks.a5kernels.bz import (
    BZSessionAdapter,
    CatlassValidationExecutor,
    ProfileCommandExecutor,
)
from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.fixtures import Language, fixture_for
from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    CollectionManifest,
    EmbeddingBackend,
    KnowledgeDB,
    SearchHit,
)
from benchmarks.a5kernels.embeddings import EmbeddingLoadError, PinnedBGEEmbeddings
from benchmarks.a5kernels.knowledge_agent import (
    Citation,
    KnowledgeAgent,
    KnowledgeQuery,
    KnowledgeResult,
    ProgressiveMemoryJournal,
    parse_knowledge_action,
)
from benchmarks.a5kernels.protocol import RunRequest, VerifiedResult
from benchmarks.a5kernels.profiling import ProfileRequest, ProfilingTreatmentController
from benchmarks.a5kernels.profiling_bz import BZProfileBackend
from benchmarks.a5kernels.runner import A5KernelRunner, ExecutionBackend
from benchmarks.a5kernels.model_profile import (
    A5Completion,
    A5ModelProfile,
    A5RunProvenance,
    complete_a5,
    load_a5_model_profile,
)

__all__ = [
    "A5KernelRunner",
    "A5Completion",
    "A5ModelProfile",
    "A5RunProvenance",
    "BZSessionAdapter",
    "BZProfileBackend",
    "CatlassValidationExecutor",
    "Citation",
    "CollectionManifest",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "EmbeddingBackend",
    "EmbeddingLoadError",
    "ExecutionBackend",
    "EvidenceKind",
    "EvidenceLedger",
    "KnowledgeAgent",
    "KnowledgeDB",
    "KnowledgeQuery",
    "KnowledgeResult",
    "Language",
    "PinnedBGEEmbeddings",
    "ProfileCommandExecutor",
    "ProfileRequest",
    "ProfilingTreatmentController",
    "ProgressiveMemoryJournal",
    "RunRequest",
    "SearchHit",
    "VerifiedResult",
    "fixture_for",
    "parse_knowledge_action",
    "complete_a5",
    "load_a5_model_profile",
]
