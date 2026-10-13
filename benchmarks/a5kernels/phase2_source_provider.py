"""Treatment-bound direct-model source generation for A5 Phase 2."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
from benchmarks.a5kernels.phase1_experiments import KnowledgeMode, ProfilingGuidance
from benchmarks.a5kernels.phase1_provider import (
    DirectCompletionProvider,
    direct_profile,
)
from benchmarks.a5kernels.phase2_attention import (
    ATTENTION_CASES,
    CANDIDATE_ABI,
    CANDIDATE_ABI_SHA256,
    HOST_TARGET_GRADING_CONTRACT,
    HOST_TARGET_GRADING_SHA256,
    SUITE_SHA256,
    TARGET_ID,
)
from benchmarks.a5kernels.phase2_live import Phase2AttemptRequest


PROVIDER_SCHEMA = "a5-catlass-phase2-source-provider-v1"
_SYSTEM_PROMPT = """You write one Catlass DSL FlashAttention candidate for host testing.
The host owns all correctness decisions. Use exactly one JSON object as each response:
{"knowledge_query":{"query":"...","limit":5}} to request gated references, or
{"candidate_source":{"source":"complete Python source"}} to submit a candidate.
Do not wrap the JSON in Markdown and do not include prose outside the JSON object."""
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_MAX_CONTEXT_BYTES = 512 * 1024
_MAX_GUIDE_BYTES = 256 * 1024


KnowledgeFactory = Callable[[Path, object], object]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError("completed provider evidence conflicts") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_context(value: object, label: str) -> tuple[bytes, str]:
    try:
        payload = canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if len(payload) > _MAX_CONTEXT_BYTES:
        raise ValueError(f"{label} exceeds the provider context bound")
    return payload, _sha256(payload)


def _parse_action(text: str) -> tuple[str, object] | None:
    """Accept one complete JSON action, never an object embedded in prose."""

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or len(value) != 1:
        return None
    if set(value) == {"knowledge_query"}:
        payload = value["knowledge_query"]
        if not isinstance(payload, dict) or not set(payload) <= {"query", "limit"}:
            return None
        try:
            return "knowledge_query", KnowledgeQuery(
                payload.get("query"), payload.get("limit", 5)
            )
        except (TypeError, ValueError):
            return None
    if set(value) == {"candidate_source"}:
        payload = value["candidate_source"]
        if not isinstance(payload, dict) or set(payload) != {"source"}:
            return None
        source = payload["source"]
        if not isinstance(source, str) or not source.strip():
            return None
        return "candidate_source", source
    return None


def _target_contract() -> dict[str, object]:
    """Serialize the checked host target without duplicating its case values."""

    candidate_abi = dict(CANDIDATE_ABI)
    return {
        "target_id": TARGET_ID,
        "suite_sha256": SUITE_SHA256,
        "candidate_abi_sha256": CANDIDATE_ABI_SHA256,
        "candidate_abi": candidate_abi,
        "host_target_grading_sha256": HOST_TARGET_GRADING_SHA256,
        "host_target_grading": dict(HOST_TARGET_GRADING_CONTRACT),
        "candidate_module": {
            "only_import": candidate_abi["module_import"],
            "module_members": candidate_abi["module_members"],
            "only_decorator": candidate_abi["kernel_decorator"],
            "only_export": candidate_abi["kernel_export"],
            "signature_policy": candidate_abi["signature_policy"],
            "host_binds_compile_time_constants": list(
                candidate_abi["host_bound_constants"]
            ),
            "host_bound_dtype": candidate_abi["host_bound_dtype"],
            "candidate_assignment_policy": candidate_abi[
                "candidate_assignment_policy"
            ],
            "launch_policy": {
                "block_num": candidate_abi["launch_block_num"],
                "required_coverage": candidate_abi["launch_coverage"],
            },
        },
        "semantics": {
            "tensor_arguments": (
                "gm_q/gm_k/gm_v/gm_output are flat contiguous row-major tensors "
                "representing logical [B,H,S,D]"
            ),
            "logical_indexing": (
                "use BATCH/HEADS/SEQUENCE/HEAD_DIM compile-time constants"
            ),
            "evaluation": (
                "candidate source is evaluated unchanged; unsupported Catlass "
                "constructs surface as host compile/runtime evidence"
            ),
            "operation": "dense forward-prefill attention",
            "dtypes": ["float16", "bfloat16"],
            "masks": ["noncausal", "causal"],
            "scale": "1/sqrt(D)",
            "result": "attention output only",
        },
        "cases": [
            {
                **asdict(case),
                "shape": list(case.shape),
                "scale": case.scale,
                "rtol": case.rtol,
                "atol": case.atol,
            }
            for case in ATTENTION_CASES
        ],
        "excluded": [
            "backward", "dropout", "KV cache", "GQA", "paging", "RoPE",
            "ALiBi", "packed variable-length inputs",
        ],
    }


class Phase2SourceProvider:
    """Generate candidate source without granting models correctness authority."""

    def __init__(
        self,
        provider: DirectCompletionProvider,
        *,
        evidence_root: str | Path,
        guide_text: str,
        knowledge_factory: KnowledgeFactory | None = None,
        profiling_guidance: str | None = None,
        max_turns: int = 12,
        max_queries: int = 4,
        max_completion_bytes: int = 256 * 1024,
    ) -> None:
        if not isinstance(provider, DirectCompletionProvider):
            raise TypeError("provider must be a DirectCompletionProvider")
        if not isinstance(guide_text, str) or not guide_text.strip():
            raise ValueError("programming guide text must be non-empty")
        if len(guide_text.encode("utf-8")) > _MAX_GUIDE_BYTES:
            raise ValueError("programming guide exceeds the provider context bound")
        if knowledge_factory is not None and not callable(knowledge_factory):
            raise TypeError("knowledge_factory must be callable")
        if type(max_turns) is not int or not 1 <= max_turns <= 40:
            raise ValueError("max_turns must be in [1, 40]")
        if type(max_queries) is not int or not 0 <= max_queries <= max_turns:
            raise ValueError("max_queries must be in [0, max_turns]")
        if (
            type(max_completion_bytes) is not int
            or not 1024 <= max_completion_bytes <= 1024 * 1024
        ):
            raise ValueError("max_completion_bytes must be in [1024, 1048576]")
        self.provider = provider
        self.evidence_root = Path(evidence_root).resolve()
        self.guide_text = guide_text
        self.knowledge_factory = knowledge_factory
        self.profiling_guidance = profiling_guidance
        self.max_turns = max_turns
        self.max_queries = max_queries
        self.max_completion_bytes = max_completion_bytes

    def __call__(self, request: Phase2AttemptRequest) -> str:
        if not isinstance(request, Phase2AttemptRequest):
            raise TypeError("request must be a Phase2AttemptRequest")
        self._validate_request(request)
        prompt, binding = self._prompt_and_binding(request)
        root = self._attempt_root(request)
        retained = self._resume(root, binding)
        if retained is not None:
            return retained

        profile = direct_profile(request.cell.model)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        transcript: list[dict[str, object]] = []
        queries: list[dict[str, object]] = []
        tokens = 0
        knowledge = self._knowledge(root, request)
        try:
            for turn in range(1, self.max_turns + 1):
                completion = self.provider.complete(profile, messages)
                tokens += completion.completion_tokens
                payload = completion.text.encode("utf-8")
                summary: dict[str, object] = {
                    "turn": turn,
                    "completion_sha256": _sha256(payload),
                    "completion_bytes": len(payload),
                    "completion_tokens": completion.completion_tokens,
                }
                if len(payload) > self.max_completion_bytes:
                    action = None
                    summary["action"] = "oversized"
                else:
                    action = _parse_action(completion.text)
                    summary["action"] = action[0] if action else "invalid"
                messages.append({"role": "assistant", "content": completion.text})

                if action is not None and action[0] == "candidate_source":
                    source = action[1]
                    assert isinstance(source, str)
                    if len(source.encode("utf-8")) > self.max_completion_bytes:
                        action = None
                        summary["action"] = "oversized"
                if action is not None and action[0] == "candidate_source":
                    source = action[1]
                    assert isinstance(source, str)
                    summary["candidate_sha256"] = _sha256(source.encode("utf-8"))
                    transcript.append(summary)
                    self._commit(
                        root, source, binding, profile.identity.as_dict(),
                        tokens, transcript, queries,
                    )
                    return source

                if action is not None and action[0] == "knowledge_query":
                    query = action[1]
                    assert isinstance(query, KnowledgeQuery)
                    if len(queries) >= self.max_queries:
                        observation = {"knowledge_query": {
                            "available": knowledge is not None,
                            "error": "query-budget-exhausted",
                            "results": [],
                        }}
                    elif knowledge is None:
                        observation = {
                            "knowledge_query": {"available": False, "results": []}
                        }
                        queries.append({
                            "query": query.query, "limit": query.limit,
                            "available": False, "citations": [],
                        })
                    else:
                        results = knowledge.query(query)
                        citations = [asdict(result.citation) for result in results]
                        observation = {"knowledge_query": {
                            "available": True,
                            "results": [
                                {"citation": citation, "text": result.text}
                                for citation, result in zip(citations, results)
                            ],
                        }}
                        queries.append({
                            "query": query.query, "limit": query.limit,
                            "available": True, "citations": citations,
                        })
                else:
                    observation = {"host": {
                        "event": "invalid-action",
                        "required": "one knowledge_query or candidate_source JSON object",
                    }}
                observation_text = json.dumps(
                    observation, sort_keys=True, separators=(",", ":")
                )
                summary["observation_sha256"] = _sha256(observation_text.encode())
                transcript.append(summary)
                messages.append({"role": "user", "content": observation_text})
        finally:
            close = getattr(knowledge, "close", None)
            if callable(close):
                close()
        raise RuntimeError("Phase 2 source provider turn budget exhausted")

    def _validate_request(self, request: Phase2AttemptRequest) -> None:
        if request.cell.profiling is ProfilingGuidance.NEW:
            raise ValueError("pending profiling treatment cannot enter source provider")
        if request.cell.profiling is ProfilingGuidance.WITH_GUIDANCE:
            if not isinstance(self.profiling_guidance, str) or not self.profiling_guidance.strip():
                raise ValueError("profiling-guidance treatment requires non-empty guidance")
        elif self.profiling_guidance is not None:
            raise ValueError("without-guidance treatment must not receive guidance")
        if request.cell.knowledge is KnowledgeMode.WITH_KDB and self.knowledge_factory is None:
            raise ValueError("with-KDB treatment requires a Knowledge Agent factory")
        if _SHA256.fullmatch(request.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a SHA-256 digest")
        if type(request.attempt) is not int or request.attempt < 1:
            raise ValueError("attempt must be positive")
        if any(
            _IDENTIFIER.fullmatch(value) is None
            for value in (request.request_id, request.attempt_id)
        ):
            raise ValueError("request and attempt IDs must be safe identifiers")
        if len(request.phase1_memory) != 8:
            raise ValueError("Phase 1 memory must contain all eight projects")
        if len(request.phase2_memory) > 3:
            raise ValueError("Phase 2 memory exceeds the semantic-attempt bound")
        if request.attempt != len(request.phase2_memory) + 1:
            raise ValueError("Phase 2 memory does not precede this attempt")

    def _prompt_and_binding(
        self, request: Phase2AttemptRequest,
    ) -> tuple[str, dict[str, object]]:
        guide_sha256 = _sha256(self.guide_text.encode("utf-8"))
        if guide_sha256 != request.cell.guide.guide_sha256:
            raise ValueError("programming guide text changed its registered identity")
        _phase1, phase1_sha256 = _canonical_context(
            list(request.phase1_memory), "Phase 1 memory"
        )
        _phase2, phase2_sha256 = _canonical_context(
            list(request.phase2_memory), "Phase 2 memory"
        )
        guidance_sha256 = (
            _sha256(self.profiling_guidance.encode("utf-8"))
            if self.profiling_guidance is not None else None
        )
        target_contract = _target_contract()
        if target_contract["suite_sha256"] != request.cell.suite_sha256:
            raise ValueError("host target contract changed its registered suite")
        target_contract_sha256 = canonical_digest(target_contract)
        prompt_value = {
            "task": "Produce Catlass DSL source for the host-owned attention suite.",
            "host_verification_only": True,
            "cell": request.cell.as_dict(),
            "protocol_sha256": request.protocol_sha256,
            "request_id": request.request_id,
            "attempt_id": request.attempt_id,
            "attempt": request.attempt,
            "target_suite_sha256": request.cell.suite_sha256,
            "target_contract": target_contract,
            "programming_guide": {
                "identity": request.cell.guide.as_dict(), "text": self.guide_text,
            },
            "phase1_memory": list(request.phase1_memory),
            "phase2_memory": list(request.phase2_memory),
            "knowledge_access": request.cell.knowledge.value,
            "profiling_guidance": self.profiling_guidance,
        }
        prompt_bytes, prompt_sha256 = _canonical_context(prompt_value, "provider prompt")
        binding = {
            "cell_id": request.cell.cell_id,
            "protocol_sha256": request.protocol_sha256,
            "request_id": request.request_id,
            "attempt_id": request.attempt_id,
            "attempt": request.attempt,
            "provider": request.cell.model.as_dict(),
            "prompt_sha256": prompt_sha256,
            "phase1_memory_sha256": phase1_sha256,
            "phase2_memory_sha256": phase2_sha256,
            "programming_guide_sha256": guide_sha256,
            "profiling_guidance_sha256": guidance_sha256,
            "target_contract_sha256": target_contract_sha256,
        }
        return prompt_bytes.decode("utf-8"), binding

    def _attempt_root(self, request: Phase2AttemptRequest) -> Path:
        return self.evidence_root / request.cell.cell_id / request.attempt_id

    def _knowledge(self, root: Path, request: Phase2AttemptRequest):
        if request.cell.knowledge is KnowledgeMode.WITHOUT_KDB:
            return None
        assert self.knowledge_factory is not None
        knowledge = self.knowledge_factory(root / "knowledge.jsonl", request.cell)
        if getattr(knowledge, "enabled", None) is not True or not callable(
            getattr(knowledge, "query", None)
        ):
            raise TypeError("Knowledge Agent factory returned an invalid enabled gate")
        return knowledge

    @staticmethod
    def _commit(
        root: Path, source: str, binding: Mapping[str, object],
        provider_identity: Mapping[str, object], tokens: int,
        transcript: list[dict[str, object]], queries: list[dict[str, object]],
    ) -> None:
        source_bytes = source.encode("utf-8")
        body = {
            "schema": PROVIDER_SCHEMA,
            **dict(binding),
            "provider": dict(provider_identity),
            "completion_tokens": tokens,
            "turns": len(transcript),
            "knowledge_queries": queries,
            "transcript": transcript,
            "candidate_sha256": _sha256(source_bytes),
        }
        manifest = {**body, "record_sha256": canonical_digest(body)}
        _publish_exact(root / "candidate.py", source_bytes)
        _publish_exact(root / "provider.json", canonical_bytes(manifest) + b"\n")

    @staticmethod
    def _resume(root: Path, binding: Mapping[str, object]) -> str | None:
        source_path = root / "candidate.py"
        manifest_path = root / "provider.json"
        if not source_path.exists() and not manifest_path.exists():
            return None
        if source_path.exists() != manifest_path.exists():
            # Candidate and binding manifest commit as a pair.  A process can
            # die after publishing either attempt-local file; neither half is
            # committed evidence.  Delete only that partial file and replay
            # this same request from the last committed semantic state.
            partial = source_path if source_path.exists() else manifest_path
            if not partial.is_file():
                raise ValueError("completed provider evidence is incomplete")
            partial.unlink()
            return None
        if not source_path.is_file() or not manifest_path.is_file():
            raise ValueError("completed provider evidence is incomplete")
        try:
            source = source_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("completed provider evidence is invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("completed provider evidence is invalid")
        digest = manifest.pop("record_sha256", None)
        if (
            manifest.get("schema") != PROVIDER_SCHEMA
            or digest != canonical_digest(manifest)
            or any(manifest.get(key) != value for key, value in binding.items())
            or manifest.get("candidate_sha256") != _sha256(source.encode("utf-8"))
            or not source.strip()
        ):
            raise ValueError("completed provider evidence changed")
        return source


__all__ = ["PROVIDER_SCHEMA", "Phase2SourceProvider"]
