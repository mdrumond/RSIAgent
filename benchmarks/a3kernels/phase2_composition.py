"""Concrete provider, knowledge, and BZ execution composition for A3 Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from benchmarks.a3_experiments import (
    A3ExperimentCell, KnowledgeMode, ProfilingGuidance,
)
from benchmarks.a3_model_profiles import A3Completion, load_a3_model_profile
from benchmarks.a3kernels.candidate import A3CandidateBackend
from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
from benchmarks.a3kernels.live_composition import (
    LiveComposition, LiveDependencies, RemoteCandidateBundle,
)
from benchmarks.a3kernels.phase1_registry import (
    DEFAULT_PROPOSALS, CurriculumProposal,
)
from benchmarks.a3kernels.phase1_wave import CellPaths, Phase1Wave
from benchmarks.a3kernels.phase2_live import (
    Phase2LiveDependencies, PracticeExecution, phase2_cells,
)
from benchmarks.a3kernels.phase2_memory import Phase1Snapshot
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2Identity, A3Phase2InfrastructureError, CurriculumDecision,
    GroundedLearning, GroundedTarget,
)
from benchmarks.a3kernels.phase2_target import Phase2TargetVerifier


_SYSTEM = """You are an A3 Ascend C research agent. Return exactly the requested
JSON schema. Host evidence alone determines PASS or FAIL. Never claim a semantic
verdict. Use only ordinary, reusable tiled vector-add techniques."""


def qualification_cells() -> tuple[A3ExperimentCell, ...]:
    """One no-KDB/no-profile qualification cell for each provider."""
    return tuple(
        cell for cell in phase2_cells()
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB
        and cell.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )


def _json_object(text: str, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


class Phase2Composition:
    """Build treatment-bound callbacks entirely from the Phase 1 live stack."""

    def __init__(
        self, *, config, dependencies: LiveDependencies,
        execution_profile: str,
    ) -> None:
        if execution_profile not in {"bz-a3-1", "bz-a3-2"}:
            raise ValueError("Phase 2 requires a BZ-A3 execution profile")
        if dependencies.execution_profile != execution_profile:
            raise ValueError("live dependencies changed the execution profile")
        self.config = config
        self.live = dependencies
        self.execution_profile = execution_profile

    def dependencies(
        self, cell: A3ExperimentCell, snapshot: Phase1Snapshot, root: Path,
    ) -> Phase2LiveDependencies:
        if cell not in phase2_cells():
            raise ValueError("Phase 2 composition requires a tiled cell")
        root = Path(root).resolve()
        paths = CellPaths(
            root, root / "workspace", root / "practice-memory.jsonl",
            root / "practice-evidence.jsonl", root / "practice-terminal.json",
            root / "workspace" / "a3_profile_driver.py",
        )
        profile = load_a3_model_profile(cell.backend_model)
        actor = self.live.actor_factory(cell, DEFAULT_PROPOSALS[0])
        knowledge = self.live.knowledge_factory(cell, paths)
        candidate = self.live.candidate_factory(cell, DEFAULT_PROPOSALS[0], paths)
        backend = (
            candidate.backend
            if isinstance(candidate, RemoteCandidateBundle)
            else candidate
        )
        verifier = Phase2TargetVerifier(
            backend, plan_builder=A3CandidateBackend(lambda *_a, **_k: None).plan,
        )

        def complete(payload: Mapping[str, object]) -> dict[str, object]:
            completion = actor(
                profile,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
            if not isinstance(completion, A3Completion):
                raise TypeError("Phase 2 actor must return A3Completion")
            if completion.provenance.get("profile_sha256") != profile.fingerprint:
                raise ValueError("Phase 2 completion changed model treatment")
            return _json_object(completion.text, label="Phase 2 action")

        def target_source(
            identity: A3Phase2Identity, memory: dict[str, object], attempt: int,
        ) -> str:
            payload: dict[str, object] = {
                "schema": "a3-phase2-target-actor-v1", "role": "target-actor",
                "identity": identity.evidence_fields(), "attempt": attempt,
                "profiling_treatment": cell.profiling.value,
                "memory": memory,
                "actions": ["source", "query"],
                "source_schema": {"action": "source", "source": "Ascend C++"},
                "query_schema": {"action": "query", "query": "text"},
            }
            action = complete(payload)
            if set(action) == {"action", "query"} and action["action"] == "query":
                query = KnowledgeQuery(action["query"])
                results = knowledge.query(query)
                payload["knowledge"] = [
                    {
                        "citation": {
                            name: getattr(item.citation, name)
                            for name in (
                                "collection", "path", "start_line", "end_line",
                                "chunk_id", "source_revision", "content_sha256",
                            )
                        },
                        "text": item.text,
                    }
                    for item in results[:5]
                ]
                payload["actions"] = ["source"]
                action = complete(payload)
            if set(action) != {"action", "source"} or action["action"] != "source":
                raise ValueError("target actor action has an invalid schema")
            source = action["source"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("target actor source must be non-empty")
            return source

        def target_verify(source, request_id, attempt_id, workdir):
            return verifier.verify(
                source, request_id=request_id, attempt_id=attempt_id,
                execution_profile=self.execution_profile, workdir=workdir,
            )

        def curriculum(
            identity: A3Phase2Identity, target: GroundedTarget,
            learning: GroundedLearning, memory: dict[str, object], evolution: int,
        ) -> CurriculumDecision:
            action = complete({
                "schema": "a3-phase2-curriculum-v1", "role": "curriculum",
                "identity": identity.evidence_fields(), "host_target": {
                    "verdict": target.verdict.value, "report": target.report,
                    "evidence_sha256": target.evidence_sha256,
                },
                "grounded_learning": learning.diagnosis, "memory": memory,
                "evolution": evolution,
                "ready_schema": {"action": "ready", "reason": "text"},
                "practice_schema": {
                    "action": "practice", "reason": "text",
                    "proposal": DEFAULT_PROPOSALS[0].as_dict(),
                },
            })
            if set(action) == {"action", "reason"} and action["action"] == "ready":
                return CurriculumDecision.ready(identity, action["reason"])
            if (
                set(action) == {"action", "reason", "proposal"}
                and action["action"] == "practice"
            ):
                return CurriculumDecision.practice(
                    identity, CurriculumProposal.from_mapping(action["proposal"]),
                    action["reason"],
                )
            raise ValueError("curriculum action has an invalid schema")

        def practice(
            _identity: A3Phase2Identity, proposal: CurriculumProposal,
            _memory: dict[str, object], evolution: int,
        ) -> PracticeExecution:
            practice_root = root / "practice" / f"{evolution}-{proposal.project_id}"
            practice_paths = CellPaths(
                practice_root, practice_root / "workspace",
                practice_root / "memory.jsonl", practice_root / "evidence.jsonl",
                practice_root / "terminal.json",
                practice_root / "workspace" / "a3_profile_driver.py",
            )
            Phase1Wave(self.config, lambda *_: {})._stage(practice_paths)
            outcome = LiveComposition(
                self.config, self.live, proposals=(proposal,),
                lineage_prefix=f"phase2-practice-{evolution}",
            ).execute(cell, practice_paths)
            if outcome["status"] != "passed":
                if outcome["terminal_reason"] == "infrastructure-unverified":
                    raise A3Phase2InfrastructureError("practice infrastructure unavailable")
                raise RuntimeError("focused Phase 2 practice did not complete")
            entry = json.loads(practice_paths.memory.read_text().splitlines()[-1])
            memory = entry["memory"]
            fact = next(
                item for item in memory["host_facts"]
                if item["category"] == "host-verification" and item["success"]
            )
            return PracticeExecution(
                proposal.project_id, memory["source_revision"],
                fact["evidence_sha256"], True,
                "focused practice completed with host verification",
            )

        return Phase2LiveDependencies(
            cell, self.execution_profile, target_source, target_verify,
            curriculum, practice,
        )


__all__ = ["Phase2Composition", "qualification_cells"]
