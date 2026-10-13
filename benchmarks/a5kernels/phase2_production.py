"""Fail-closed production composition for one admitted A5 Phase 2 cell."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Mapping

from benchmarks.a5kernels.bz import CatlassValidationExecutor
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, ProgressiveMemoryJournal
from benchmarks.a5kernels.phase1_composition import (
    A5_PROFILING_GUIDANCE,
    LiveDependencies,
    Phase1CellComposition,
)
from benchmarks.a5kernels.phase1_experiments import (
    KnowledgeMode,
    ProfilingGuidance,
    admit_programming_guide,
)
from benchmarks.a5kernels.phase1_live import GateReport
from benchmarks.a5kernels.phase1_production import load_direct_environment
from benchmarks.a5kernels.phase1_provider import (
    DirectCompletionProvider,
    Transport,
    openai_compatible_transport,
)
from benchmarks.a5kernels.phase2_attention import (
    ATTENTION_CASES,
    AttentionTargetVerifier,
    build_execution,
)
from benchmarks.a5kernels.phase2_attention_bz import (
    ATTENTION_CANDIDATE_FIXTURE,
    CatlassAttentionBackend,
)
from benchmarks.a5kernels.phase2_live import Phase2Dependencies
from benchmarks.a5kernels.phase2_memory import (
    A5Phase1Snapshot,
    admit_phase1_snapshot,
    phase2_cells,
)
from benchmarks.a5kernels.phase2_source_provider import Phase2SourceProvider
from benchmarks.a5kernels.production import _probe_knowledge_query, _transformers_cache


QUALIFICATION_SCHEMA = "a5-catlass-phase2-transport-qualification-v1"


@dataclass(frozen=True)
class Phase2ProductionConfig:
    """Machine-local production inputs for one exact Phase 2 destination."""

    phase1_root: Path
    gate: GateReport
    guide_path: Path
    phase2_root: Path
    snapshot_path: Path
    cell_id: str
    tla_root: Path
    catlass_source: str
    device: int
    runtime_catlass_revision: str | None = None
    bge_cache: Path | None = None
    kdb: Path | None = None
    collection: str | None = None
    env_file: Path | None = None
    phase2_guide_path: Path | None = None
    phase2_cpl_skills_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gate, GateReport):
            raise TypeError("gate must be a GateReport")
        if not Path(self.catlass_source).is_absolute():
            raise ValueError("catlass_source must be an absolute retained BZ path")
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("device must be a non-negative integer")
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if self.runtime_catlass_revision is not None and (
            len(self.runtime_catlass_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.runtime_catlass_revision)
        ):
            raise ValueError("runtime_catlass_revision must be a lowercase 40-character SHA")
        if (self.phase2_guide_path is None) != (
            self.phase2_cpl_skills_revision is None
        ):
            raise ValueError(
                "Phase 2 guide path and CPL skills revision must be supplied together"
            )

    @property
    def validation_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/catlass-validation.sh"

    @property
    def session_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/session.sh"

    @property
    def upload_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/upload.sh"


def _unavailable_dependencies() -> LiveDependencies:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Phase 1 composition is read-only during Phase 2 admission")

    return LiveDependencies(unavailable, unavailable)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError("immutable Phase 2 qualification conflicts") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _qualification_identity(config: Phase2ProductionConfig, cell, provenance):
    case = ATTENTION_CASES[0]
    execution = build_execution(case)
    return execution, {
        "schema": QUALIFICATION_SCHEMA,
        "cell_id": cell.cell_id,
        "catlass_source": config.catlass_source,
        "catlass_revision": cell.catlass_revision,
        "runtime_catlass_revision": dict(provenance)["catlass_revision"],
        "runtime_provenance": [list(item) for item in provenance],
        "physical_device": config.device,
        "case": asdict(case),
        "case_sha256": canonical_digest(asdict(case)),
        "candidate_sha256": _sha(ATTENTION_CANDIDATE_FIXTURE.encode("utf-8")),
        "input_sha256": canonical_digest({
            "q": execution.q, "k": execution.k, "v": execution.v,
            "scale": execution.scale,
        }),
    }


def _validate_qualification(value, identity, execution) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid Phase 2 transport qualification")
    retained = dict(value)
    digest = retained.pop("record_sha256", None)
    output = retained.get("output")
    if (
        digest != canonical_digest(retained)
        or any(retained.get(key) != expected for key, expected in identity.items())
        or retained.get("status") != "transport-qualified"
        or retained.get("exit_code") != 0
        or not isinstance(retained.get("session_handle"), str)
        or not retained["session_handle"].startswith("bz-a5:")
        or not isinstance(output, list)
        or len(output) != len(execution.v)
        or retained.get("output_sha256") != canonical_digest(output)
    ):
        raise ValueError("invalid Phase 2 transport qualification")
    tolerance = ATTENTION_CASES[0].atol
    if any(
        isinstance(got, bool) or not isinstance(got, (int, float))
        or not math.isfinite(got)
        or abs(float(got) - expected) > tolerance + tolerance * abs(expected)
        for got, expected in zip(output, execution.v)
    ):
        raise ValueError("transport fixture did not reproduce input V")
    return value


def _qualify(config, cell, backend, provenance) -> Mapping[str, object]:
    path = config.phase2_root.resolve() / "cells" / cell.cell_id / "qualification.json"
    execution, identity = _qualification_identity(config, cell, provenance)
    if path.exists():
        try:
            retained = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Phase 2 transport qualification") from exc
        return _validate_qualification(retained, identity, execution)

    receipt = backend.execute(
        ATTENTION_CANDIDATE_FIXTURE,
        execution,
        path.parent / "qualification-host",
    )
    if receipt.exit_code != 0:
        raise ValueError("transport fixture did not exit successfully")
    output = list(receipt.output)
    body = {
        **identity,
        "status": "transport-qualified",
        "exit_code": receipt.exit_code,
        "session_handle": receipt.session_handle,
        "output": output,
        "output_sha256": canonical_digest(output),
        "metadata": [list(item) for item in receipt.metadata],
    }
    value = {**body, "record_sha256": canonical_digest(body)}
    _validate_qualification(value, identity, execution)
    _publish_exact(path, canonical_bytes(value) + b"\n")
    return value


def _retained_qualification_provenance(config, cell):
    path = config.phase2_root.resolve() / "cells" / cell.cell_id / "qualification.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
        provenance = tuple(tuple(item) for item in value["runtime_provenance"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Phase 2 transport qualification") from exc
    if any(
        len(item) != 2 or not all(isinstance(part, str) for part in item)
        for item in provenance
    ):
        raise ValueError("invalid Phase 2 transport qualification")
    return provenance


def _knowledge_factory(config, embeddings):
    assert config.kdb is not None and config.collection is not None

    def factory(journal_path, _cell):
        database = KnowledgeDB.open_read_only(config.kdb, embeddings)
        try:
            return KnowledgeAgent(
                enabled=True,
                database=database,
                collection=config.collection,
                journal=ProgressiveMemoryJournal(journal_path),
            )
        finally:
            database.close()

    return factory


def _phase2_guide(config: Phase2ProductionConfig, composition):
    if config.phase2_guide_path is None:
        return composition.guide, composition.guide_text
    guide = admit_programming_guide(
        config.phase2_guide_path,
        cpl_skills_revision=config.phase2_cpl_skills_revision,
        catlass_revision=composition.guide.catlass_revision,
    )
    try:
        payload = config.phase2_guide_path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("unable to read admitted Phase 2 guide content") from exc
    if hashlib.sha256(payload).hexdigest() != guide.guide_sha256:
        raise ValueError("Phase 2 guide content changed during admission")
    return guide, text


def build_phase2_production(
    config: Phase2ProductionConfig,
    *,
    environment: Mapping[str, str] | None = None,
    transport: Transport = openai_compatible_transport,
    embeddings=None,
    executor=None,
    backend=None,
) -> tuple[A5Phase1Snapshot, Phase2Dependencies]:
    """Validate local identity, qualify transport, then expose live dependencies."""

    composition = Phase1CellComposition(
        config.phase1_root,
        config.guide_path,
        config.gate.guide,
        config.gate,
        _unavailable_dependencies(),
    )
    phase2_guide, phase2_guide_text = _phase2_guide(config, composition)
    destinations = {
        cell.cell_id: cell
        for cell in phase2_cells(
            config.gate.guide,
            config.gate.report_sha256,
            destination_guide=phase2_guide,
        )
    }
    try:
        cell = destinations[config.cell_id]
    except KeyError:
        raise ValueError("requested cell is not a registered Phase 2 destination") from None
    expected_snapshot = (
        config.phase2_root.resolve() / "cells" / cell.cell_id / "snapshot"
    )
    if config.snapshot_path.resolve() != expected_snapshot:
        raise ValueError("requested snapshot path does not match the registered cell")
    snapshot = admit_phase1_snapshot(composition, cell, config.phase2_root)

    # Exhaust every local dependency check before probing or dispatching BZ-A5.
    # Constructing the provider is side-effect free; its first model call remains
    # gated by the transport qualification below.
    values = load_direct_environment(environment, env_file=config.env_file)
    provider = DirectCompletionProvider(values, transport=transport)
    if (
        transport is openai_compatible_transport
        and importlib.util.find_spec("openai") is None
    ):
        raise RuntimeError("the openai package is required for direct providers")
    knowledge_factory = None
    if cell.knowledge is KnowledgeMode.WITH_KDB:
        if config.bge_cache is None or config.kdb is None or not config.collection:
            raise ValueError("with-KDB treatment requires cache, KDB, and collection")
        if not config.bge_cache.is_dir() or not config.kdb.is_file():
            raise ValueError("pinned embeddings and KDB are unavailable")
        if embeddings is None:
            from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings

            embeddings = PinnedBGEEmbeddings(
                cache_dir=_transformers_cache(config.bge_cache)
            )
        embeddings.embed_query(["A5 Phase 2 preflight"])
        database = KnowledgeDB.open_read_only(config.kdb, embeddings)
        try:
            manifest = database.manifest(config.collection)
            if (
                manifest.embedding_model != embeddings.model
                or manifest.embedding_revision != embeddings.revision
            ):
                raise ValueError("KDB manifest does not match pinned embeddings")
            _probe_knowledge_query(database, config.collection)
        finally:
            database.close()
        knowledge_factory = _knowledge_factory(config, embeddings)

    if executor is None:
        for wrapper in (
            config.validation_wrapper, config.session_wrapper, config.upload_wrapper,
        ):
            if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
                raise ValueError("checked BZ-A5 execution wrappers are unavailable")
        executor = CatlassValidationExecutor(
            upload_wrapper=str(config.upload_wrapper),
            validation_wrapper=str(config.validation_wrapper),
            catlass_source=config.catlass_source,
            catlass_revision=(
                config.runtime_catlass_revision or cell.catlass_revision
            ),
        )
    retained_provenance = _retained_qualification_provenance(config, cell)
    live_provenance = tuple(executor.runtime_provenance)
    is_resume = retained_provenance is not None
    if is_resume and retained_provenance != live_provenance:
        raise ValueError(
            "retained Phase 2 qualification runtime provenance changed"
        )
    provenance = live_provenance
    expected_runtime_revision = (
        config.runtime_catlass_revision or cell.catlass_revision
    )
    if dict(provenance).get("catlass_revision") != expected_runtime_revision:
        raise ValueError("runtime provenance changed the configured Catlass revision")
    if dict(provenance).get("catlass_source") != config.catlass_source:
        raise ValueError("runtime provenance changed the retained Catlass source")
    if not is_resume:
        executor.probe_device(config.device)
    if backend is None:
        backend = CatlassAttentionBackend(
            executor, session_wrapper=str(config.session_wrapper), device=config.device
        )
    _qualify(config, cell, backend, provenance)

    guidance = (
        A5_PROFILING_GUIDANCE
        if cell.profiling is ProfilingGuidance.WITH_GUIDANCE else None
    )
    source = Phase2SourceProvider(
        provider,
        evidence_root=config.phase2_root.resolve() / "cells",
        guide_text=phase2_guide_text,
        knowledge_factory=knowledge_factory,
        profiling_guidance=guidance,
    )
    verifier = AttentionTargetVerifier(backend)
    return snapshot, Phase2Dependencies(
        cell,
        source,
        lambda candidate, request_id, attempt_id, workdir: verifier.verify(
            candidate,
            request_id=request_id,
            attempt_id=attempt_id,
            execution_profile="bz-a5",
            workdir=workdir,
        ),
    )


__all__ = [
    "QUALIFICATION_SCHEMA",
    "Phase2ProductionConfig",
    "build_phase2_production",
]
