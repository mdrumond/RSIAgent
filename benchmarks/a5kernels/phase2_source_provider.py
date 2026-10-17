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
from benchmarks.a5kernels.phase2_live import (
    MAX_REPAIR_SOURCE_BYTES,
    REPAIR_INSTRUCTION,
    Phase2AttemptRequest,
)


PROVIDER_SCHEMA = "a5-catlass-phase2-source-provider-v2"
_SYSTEM_PROMPT = """You write one Catlass DSL FlashAttention candidate for host testing.
The host owns all correctness decisions. Use exactly one JSON object as each response:
{"knowledge_query":{"query":"...","limit":5}} to request gated references, or
{"candidate_source":{"source":"complete Python source"}} to submit a candidate.
Do not wrap the JSON in Markdown and do not include prose outside the JSON object."""
_COMPILE_SYSTEM_PROMPT = _SYSTEM_PROMPT.replace(
    '{"candidate_source"',
    '{"compile_probe":{"source":"complete Python source"}} to request host '
    'compilation only (not execution or correctness), or\n{"candidate_source"',
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_MAX_CONTEXT_BYTES = 512 * 1024
_MAX_GUIDE_BYTES = 256 * 1024


KnowledgeFactory = Callable[[Path, object], object]
CompileProbe = Callable[[str, Path], Mapping[str, object]]
_PROBE_RESULT_KEYS = {
    "status", "case_id", "exit_code", "diagnostic", "session_handle",
}


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
    if set(value) == {"compile_probe"}:
        payload = value["compile_probe"]
        if not isinstance(payload, dict) or set(payload) != {"source"}:
            return None
        source = payload["source"]
        if not isinstance(source, str) or not source.strip():
            return None
        return "compile_probe", source
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
        compile_probe: CompileProbe | None = None,
        max_compile_probes: int = 0,
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
        if not isinstance(max_compile_probes, int) or isinstance(
            max_compile_probes, bool
        ) or not 0 <= max_compile_probes <= 20:
            raise ValueError("max_compile_probes must be in [0, 20]")
        if (compile_probe is None) != (max_compile_probes == 0):
            raise ValueError(
                "compile_probe and a positive max_compile_probes must be enabled together"
            )
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
        self.compile_probe = compile_probe
        self.max_compile_probes = max_compile_probes
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
            {"role": "system", "content": (
                _COMPILE_SYSTEM_PROMPT
                if self.compile_probe is not None else _SYSTEM_PROMPT
            )},
            {"role": "user", "content": prompt},
        ]
        transcript: list[dict[str, object]] = []
        queries: list[dict[str, object]] = []
        tokens = 0
        ordinary_turns = 0
        probes = 0
        resumed = self._resume_probe_session(root, binding)
        if resumed is not None:
            messages, transcript, queries, tokens, ordinary_turns, probes = resumed
        progress = self._resume_progress(root, binding)
        if progress is not None and len(progress[1]) >= len(transcript):
            messages, transcript, queries, tokens, ordinary_turns, probes = progress
        knowledge = self._knowledge(root, request)
        try:
            while ordinary_turns < self.max_turns:
                completion = self.provider.complete(profile, messages)
                tokens += completion.completion_tokens
                payload = completion.text.encode("utf-8")
                summary: dict[str, object] = {
                    "turn": len(transcript) + 1,
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
                    ordinary_turns += 1
                    summary["candidate_sha256"] = _sha256(source.encode("utf-8"))
                    transcript.append(summary)
                    self._commit(
                        root, source, binding, profile.identity.as_dict(),
                        tokens, transcript, queries, ordinary_turns, probes, messages,
                    )
                    return source

                if action is not None and action[0] == "compile_probe":
                    source = action[1]
                    assert isinstance(source, str)
                    if len(source.encode("utf-8")) > self.max_completion_bytes:
                        action = None
                        summary["action"] = "oversized"
                    elif self.compile_probe is None:
                        action = None
                        summary["action"] = "compile-probe-disabled"
                    elif probes >= self.max_compile_probes:
                        ordinary_turns += 1
                        observation = {"compile_probe": {
                            "status": "probe-budget-exhausted",
                            "compilation_only": True,
                            "correctness_established": False,
                        }}
                        observation_text = json.dumps(
                            observation, sort_keys=True, separators=(",", ":")
                        )
                        summary["observation_sha256"] = _sha256(
                            observation_text.encode()
                        )
                        transcript.append(summary)
                        messages.append({"role": "user", "content": observation_text})
                        self._retain_progress(
                            root, binding, messages, transcript, queries,
                            tokens, ordinary_turns, probes,
                        )
                        continue
                    else:
                        probes += 1
                        self._retain_probe_request(
                            root, probes, binding, source, messages, transcript,
                            queries, tokens, ordinary_turns, summary,
                        )
                        result = self._probe(source, root, probes, binding)
                        observation_text = self._probe_observation(
                            result, source, probes, summary
                        )
                        transcript.append(summary)
                        messages.append({"role": "user", "content": observation_text})
                        continue

                if action is not None and action[0] == "knowledge_query":
                    # Compile probes are the only accepted action exempt from
                    # the ordinary provider-turn allowance.
                    ordinary_turns += 1
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
                    ordinary_turns += 1
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
                self._retain_progress(
                    root, binding, messages, transcript, queries,
                    tokens, ordinary_turns, probes,
                )
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
        if request.attempt == 1:
            if request.repair_context is not None:
                raise ValueError("first Phase 2 attempt cannot have repair context")
        else:
            self._validate_repair_context(request)

    @staticmethod
    def _validate_repair_context(request: Phase2AttemptRequest) -> None:
        context = request.repair_context
        if not isinstance(context, Mapping):
            raise ValueError("Phase 2 retry requires preceding repair context")
        expected_keys = {
            "instruction", "prior_attempt", "candidate_sha256",
            "evidence_sha256", "memory_entry_sha256", "host_evidence",
            "candidate_source", "candidate_source_bytes",
            "candidate_source_truncated", "candidate_context_sha256",
        }
        if set(context) != expected_keys:
            raise ValueError("Phase 2 repair context has invalid fields")
        prior = request.phase2_memory[-1]
        source = context["candidate_source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Phase 2 repair context requires candidate source")
        source_payload = source.encode("utf-8")
        if len(source_payload) > MAX_REPAIR_SOURCE_BYTES:
            raise ValueError("Phase 2 repair candidate exceeds the context bound")
        if (
            context["prior_attempt"] != request.attempt - 1
            or context["instruction"] != REPAIR_INSTRUCTION
            or context["prior_attempt"] != prior.get("attempt")
            or context["candidate_sha256"] != prior.get("candidate_sha256")
            or context["evidence_sha256"] != prior.get("evidence_sha256")
            or context["memory_entry_sha256"] != prior.get("entry_sha256")
            or context["host_evidence"] != prior
            or context["candidate_context_sha256"] != _sha256(source_payload)
            or type(context["candidate_source_bytes"]) is not int
            or context["candidate_source_bytes"] < len(source_payload)
            or type(context["candidate_source_truncated"]) is not bool
            or context["candidate_source_truncated"]
            != (context["candidate_source_bytes"] > len(source_payload))
        ):
            raise ValueError("Phase 2 repair context changed preceding evidence")
        if (
            not context["candidate_source_truncated"]
            and _sha256(source_payload) != context["candidate_sha256"]
        ):
            raise ValueError("Phase 2 repair source changed preceding candidate")

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
        _repair, repair_sha256 = _canonical_context(
            request.repair_context, "Phase 2 repair context"
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
        if self.compile_probe is not None:
            prompt_value["compile_probes"] = {
                "enabled": True,
                "allowance": self.max_compile_probes,
                "counts_against_ordinary_turns": False,
                "compilation_only": True,
                "correctness_established": False,
            }
        if request.repair_context is not None:
            prompt_value["repair_context"] = request.repair_context
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
            "repair_context_sha256": repair_sha256,
            "programming_guide_sha256": guide_sha256,
            "profiling_guidance_sha256": guidance_sha256,
            "target_contract_sha256": target_contract_sha256,
        }
        if self.compile_probe is not None:
            binding["max_compile_probes"] = self.max_compile_probes
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
        ordinary_turns: int, probes: int,
        messages: list[dict[str, str]],
    ) -> None:
        source_bytes = source.encode("utf-8")
        body = {
            "schema": PROVIDER_SCHEMA,
            **dict(binding),
            "provider": dict(provider_identity),
            "completion_tokens": tokens,
            "turns": len(transcript),
            "ordinary_turns": ordinary_turns,
            "compile_probes": probes,
            "knowledge_queries": queries,
            "transcript": transcript,
            "messages": messages,
            "candidate_sha256": _sha256(source_bytes),
        }
        manifest = {**body, "record_sha256": canonical_digest(body)}
        _publish_exact(root / "candidate.py", source_bytes)
        _publish_exact(root / "provider.json", canonical_bytes(manifest) + b"\n")

    @staticmethod
    def _retain_progress(
        root: Path, binding: Mapping[str, object],
        messages: list[dict[str, str]], transcript: list[dict[str, object]],
        queries: list[dict[str, object]], tokens: int, ordinary_turns: int,
        probes: int,
    ) -> None:
        body = {
            "attempt_binding_sha256": canonical_digest(binding),
            "messages": messages, "transcript": transcript,
            "knowledge_queries": queries, "completion_tokens": tokens,
            "ordinary_turns": ordinary_turns, "compile_probes": probes,
        }
        _publish_exact(
            root / "progress" / f"turn-{len(transcript):03d}.json",
            canonical_bytes({**body, "record_sha256": canonical_digest(body)}) + b"\n",
        )

    @staticmethod
    def _resume_progress(root: Path, binding: Mapping[str, object]):
        checkpoints = sorted((root / "progress").glob("turn-*.json"))
        if not checkpoints:
            return None
        retained = json.loads(checkpoints[-1].read_bytes())
        digest = retained.pop("record_sha256", None)
        if (
            digest != canonical_digest(retained)
            or retained.get("attempt_binding_sha256") != canonical_digest(binding)
        ):
            raise ValueError("retained provider progress changed")
        return (
            retained["messages"], retained["transcript"],
            retained["knowledge_queries"], retained["completion_tokens"],
            retained["ordinary_turns"], retained["compile_probes"],
        )

    def _probe(
        self, source: str, root: Path, ordinal: int,
        binding: Mapping[str, object],
    ) -> dict[str, object]:
        assert self.compile_probe is not None
        source_bytes = source.encode("utf-8")
        probe_root = root / "compile-probes" / f"probe-{ordinal:02d}"
        result_path = probe_root / "result.json"
        identity = {
            "schema": "a5-catlass-phase2-compile-probe-v1",
            "probe": ordinal,
            "candidate_sha256": _sha256(source_bytes),
            "attempt_binding_sha256": canonical_digest(binding),
        }
        _publish_exact(probe_root / "candidate.py", source_bytes)
        if result_path.exists():
            try:
                retained = json.loads(result_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid retained compile probe") from exc
            digest = retained.pop("record_sha256", None)
            if (
                digest != canonical_digest(retained)
                or any(retained.get(key) != value for key, value in identity.items())
            ):
                raise ValueError("retained compile probe changed identity")
            return {key: retained[key] for key in _PROBE_RESULT_KEYS}
        result = dict(self.compile_probe(source, probe_root / "host"))
        if set(result) != _PROBE_RESULT_KEYS:
            raise ValueError("compile probe returned an invalid result contract")
        if (
            result["status"] not in {"compiled", "compile-failed", "invalid-source"}
            or not isinstance(result["case_id"], str)
            or not 0 < len(result["case_id"].encode()) <= 160
            or isinstance(result["exit_code"], bool)
            or not isinstance(result["exit_code"], int)
            or not isinstance(result["diagnostic"], str)
            or len(result["diagnostic"].encode()) > 4096
            or result["session_handle"] is not None
            and (
                not isinstance(result["session_handle"], str)
                or not 0 < len(result["session_handle"].encode()) <= 256
            )
        ):
            raise ValueError("compile probe returned invalid bounded evidence")
        body = {**identity, **result}
        _publish_exact(
            result_path,
            canonical_bytes({**body, "record_sha256": canonical_digest(body)}) + b"\n",
        )
        return result

    @staticmethod
    def _probe_observation(
        result: Mapping[str, object], source: str, ordinal: int,
        summary: dict[str, object],
    ) -> str:
        candidate_sha256 = _sha256(source.encode())
        observation = {"compile_probe": {
            **dict(result),
            "probe": ordinal,
            "candidate_sha256": candidate_sha256,
            "compilation_only": True,
            "correctness_established": False,
        }}
        text = json.dumps(observation, sort_keys=True, separators=(",", ":"))
        summary["probe"] = ordinal
        summary["candidate_sha256"] = candidate_sha256
        summary["observation_sha256"] = _sha256(text.encode())
        return text

    @staticmethod
    def _retain_probe_request(
        root: Path, ordinal: int, binding: Mapping[str, object], source: str,
        messages: list[dict[str, str]], transcript: list[dict[str, object]],
        queries: list[dict[str, object]], tokens: int, ordinary_turns: int,
        summary: Mapping[str, object],
    ) -> None:
        body = {
            "schema": "a5-catlass-phase2-compile-probe-request-v1",
            "probe": ordinal,
            "attempt_binding_sha256": canonical_digest(binding),
            "candidate_sha256": _sha256(source.encode()),
            "source": source,
            "messages": messages,
            "transcript": transcript,
            "knowledge_queries": queries,
            "completion_tokens": tokens,
            "ordinary_turns": ordinary_turns,
            "summary": dict(summary),
        }
        _publish_exact(
            root / "compile-probes" / f"probe-{ordinal:02d}" / "request.json",
            canonical_bytes({**body, "record_sha256": canonical_digest(body)}) + b"\n",
        )

    def _resume_probe_session(
        self, root: Path, binding: Mapping[str, object],
    ) -> tuple[
        list[dict[str, str]], list[dict[str, object]], list[dict[str, object]],
        int, int, int,
    ] | None:
        probe_root = root / "compile-probes"
        requests = sorted(probe_root.glob("probe-*/request.json"))
        if not requests:
            return None
        try:
            retained = json.loads(requests[-1].read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid retained compile probe request") from exc
        digest = retained.pop("record_sha256", None)
        ordinal = retained.get("probe")
        source = retained.get("source")
        if (
            retained.get("schema") != "a5-catlass-phase2-compile-probe-request-v1"
            or digest != canonical_digest(retained)
            or retained.get("attempt_binding_sha256") != canonical_digest(binding)
            or not isinstance(ordinal, int)
            or ordinal != len(requests)
            or not isinstance(source, str)
            or retained.get("candidate_sha256") != _sha256(source.encode())
        ):
            raise ValueError("retained compile probe request changed")
        result = self._probe(source, root, ordinal, binding)
        messages = list(retained["messages"])
        transcript = list(retained["transcript"])
        queries = list(retained["knowledge_queries"])
        summary = dict(retained["summary"])
        observation = self._probe_observation(result, source, ordinal, summary)
        transcript.append(summary)
        messages.append({"role": "user", "content": observation})
        return (
            messages, transcript, queries, int(retained["completion_tokens"]),
            int(retained["ordinary_turns"]), ordinal,
        )

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
        Phase2SourceProvider._validate_probes(root, binding, manifest)
        return source

    @staticmethod
    def _validate_probes(
        root: Path, binding: Mapping[str, object], manifest: Mapping[str, object],
    ) -> None:
        summaries = [
            item for item in manifest.get("transcript", ())
            if isinstance(item, dict) and item.get("action") == "compile_probe"
            and isinstance(item.get("probe"), int)
        ]
        if len(summaries) != manifest.get("compile_probes", 0):
            raise ValueError("completed provider compile probes changed")
        for summary in summaries:
            ordinal = summary["probe"]
            probe_root = root / "compile-probes" / f"probe-{ordinal:02d}"
            try:
                source = (probe_root / "candidate.py").read_bytes()
                result = json.loads((probe_root / "result.json").read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("completed provider compile probe is invalid") from exc
            digest = result.pop("record_sha256", None)
            if (
                _sha256(source) != summary.get("candidate_sha256")
                or result.get("candidate_sha256") != summary.get("candidate_sha256")
                or result.get("attempt_binding_sha256") != canonical_digest(binding)
                or digest != canonical_digest(result)
            ):
                raise ValueError("completed provider compile probe changed")

    def conversation(
        self, cell_id: str, attempt_id: str,
    ) -> tuple[Mapping[str, str], ...] | None:
        """Read the exact retained provider conversation for SFT trace capture."""
        manifest = json.loads(
            (self.evidence_root / cell_id / attempt_id / "provider.json").read_bytes()
        )
        messages = manifest.get("messages")
        if messages is None:
            return None
        if not isinstance(messages, list) or not messages:
            raise ValueError("provider conversation is invalid")
        if any(
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item["role"] not in {"system", "user", "assistant"}
            or not isinstance(item["content"], str)
            for item in messages
        ):
            raise ValueError("provider conversation is invalid")
        return tuple(dict(item) for item in messages)


__all__ = ["PROVIDER_SCHEMA", "Phase2SourceProvider"]
