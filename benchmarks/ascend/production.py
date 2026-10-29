"""Target bindings for the shared adaptive controller and host verifier."""

from dataclasses import dataclass, replace
from pathlib import Path
from coding.pi.actor import PiActorDriver
from coding.pi import ENGINE_IDENTITY

import re
from queue import Queue
from threading import Lock, Condition

from benchmarks.ascend.contracts import CompileObservation, RuntimeObservation
from benchmarks.ascend.targets import a3_curriculum_policy
from benchmarks.a3kernels.native_candidate import NativeCandidatePlan, NativeTensor, native_infrastructure_failure
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.a3kernels.phase1_protocol import FailedEvidence
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_project import ProjectOp
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutionFamily, ProjectKernelABI


def native_target_runtime(config, transport):
    """Canonical goal identity from the selected native transport and observations."""
    environment = config.environment_identity or {}
    compiler = getattr(config, "compiler_version", None) or environment.get("compiler_version")
    if (not compiler or not environment.get("cann_version")
            or not hasattr(transport, "_device") or not hasattr(transport, "profile")):
        return None
    runtime = getattr(transport, "runtime", None) or getattr(transport, "runtime_identity", {}).get("runtime")
    if not runtime and transport.profile in {"local", "bz-a3-1", "bz-a3-2"}:
        runtime = "local" if transport.profile == "local" else "py311-torch"
    if not runtime:
        return None
    return {"target": "a3", "runtime": runtime, "device": str(transport._device),
            "toolchain": environment["cann_version"], "compiler": compiler,
            "product": config.target, "architecture": config.arch,
            "execution_profile": transport.profile}


def compose_adaptive_controller(
    *, root, objective, initial_memory, executor, actor, profile, provider,
    immutable_context, max_parallel=1, project_target=None, max_projects=None, max_seconds=7200,
    target_policy=None, profiling_guidance=None, query_knowledge=None,
    request_profile=None, request_timing=None, request_diagnostic=None,
    diagnostic_instructions=None, curriculum_query_knowledge=None,
    unavailable_capabilities=(),
    reviewed_snapshot=None, trace_hook=None, trace_recorder_factory=None,
    project_actor=None,
):
    """One action loop, curriculum and progressive-memory controller for both targets."""
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent

    project_actor = project_actor or AdaptiveActorProjectExecutor(
        executor, actor, profile, query_knowledge=query_knowledge,
        request_profile=request_profile, request_timing=request_timing,
        request_diagnostic=request_diagnostic, immutable_context=immutable_context,
        diagnostic_instructions=diagnostic_instructions,
        retain_semantic_failure=True,
        **({"trace_recorder_factory": trace_recorder_factory} if trace_recorder_factory is not None else {}),
    )
    curriculum = CurriculumAgent(
        lambda messages: provider.complete(profile, messages).text,
        target_policy=target_policy, profiling_guidance=profiling_guidance,
        unavailable_capabilities=unavailable_capabilities,
        query_knowledge=curriculum_query_knowledge,
    )
    if reviewed_snapshot is None:
        coordinator = AdaptivePhase1Coordinator(
            root, objective, initial_memory, curriculum, project_actor,
            max_parallel=max_parallel, project_target=project_target, max_projects=max_projects, max_seconds=max_seconds,
            allow_partial=True,
        )
    else:
        from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
        if max_parallel != 1 or max_projects is not None:
            raise ValueError("adaptive Phase 2 uses serial selection and verified project_target")
        coordinator = AdaptivePhase2Coordinator(
            root, objective, reviewed_snapshot, curriculum, project_actor,
            project_target=project_target, trace_hook=trace_hook,
        )
    return project_actor, coordinator


@dataclass(frozen=True)
class NativeProductionConfig:
    root: Path
    target: str
    arch: str
    execution_profile: str
    guide_sha256: str
    cpl_skills_revision: str
    dtypes: tuple[str, ...] = ("float32",)
    kdb: Path | None = None
    collection: str | tuple[str, ...] | list[str] | None = None
    environment_identity: dict | None = None
    profiling_mode: str | None = None
    compiler_version: str | None = None
    curriculum_kdb: bool = False
    adaptive_kdb: bool = False
    librarian_seconds: float = 90
    memory_kdb: Path | None = None
    memory_collection: str | None = None
    memory_snapshot: Path | None = None

    def __post_init__(self):
        import math
        if type(self.adaptive_kdb) is not bool:
            raise TypeError("adaptive_kdb must be boolean")
        if type(self.librarian_seconds) not in (int, float) or not math.isfinite(self.librarian_seconds) or self.librarian_seconds <= 0:
            raise ValueError("librarian_seconds must be positive finite seconds")
        memory = (self.memory_kdb, self.memory_collection, self.memory_snapshot)
        if any(item is not None for item in memory) and (not all(item is not None for item in memory) or not self.adaptive_kdb):
            raise ValueError("indexed memory requires adaptive KDB, database, collection and snapshot")
        if self.adaptive_kdb and (self.kdb is None or not self.collection):
            raise ValueError("adaptive KDB requires a configured KDB and collection")
        if type(self.curriculum_kdb) is not bool:
            raise TypeError("curriculum_kdb must be boolean")
        if self.curriculum_kdb and (self.kdb is None or not self.collection):
            raise ValueError("curriculum KDB requires a configured KDB and collection")
        if self.execution_profile not in {"bz-a3-1", "bz-a3-2", "gz-a3", "local"}:
            raise ValueError("native production requires an A3 or local profile")
        if not self.target or not self.arch:
            raise ValueError("native production requires discovered target and architecture")
        if re.fullmatch(r"[0-9a-f]{64}", self.guide_sha256) is None:
            raise ValueError("native guide identity must be SHA-256")
        if re.fullmatch(r"[0-9a-f]{40}", self.cpl_skills_revision) is None:
            raise ValueError("native guide requires an exact CPL skills revision")
        a3_curriculum_policy(dtypes=self.dtypes)


@dataclass(frozen=True)
class NativeProjectWorker:
    """One exclusively assigned device's transport, evidence, and measurements."""

    root: Path
    transport: object
    request_profile: object = None
    request_timing: object = None


class NativeProjectPool:
    manages_actor_time = True
    """Exclusive slots; quarantine uncertain workers without poisoning healthy peers."""

    def __init__(self, actors, *, stopped=lambda: False):
        self.stopped = stopped
        self.actors = Queue()
        self.failure = None
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.live = 0
        self.quarantined = []
        for actor in actors:
            self.actors.put(actor)
            self.live += 1

    def __call__(self, request):
        from benchmarks.a5kernels.phase1_wave_runtime import (
            ProjectAdmissionTimeout, SemanticProjectFailure,
        )
        if getattr(request, 'remaining_seconds', None) is not None and request.remaining_seconds <= 0:
            raise TimeoutError('actor work allocation exhausted before worker admission')
        from .actor_time import preparation
        from .actor_control import ActorControl
        branch = getattr(request, 'branch_root', None)
        control = ActorControl(Path(branch) / 'actor' if branch is not None else None)
        with preparation('waiting for a native worker'):
            with self.condition:
                while True:
                    if self.stopped() or control.status()['stop_requested']:
                        error = TimeoutError('operator stopped native worker admission')
                        error.interruption_reason = 'operator-stop'
                        raise error
                    if not self.actors.empty():
                        actor = self.actors.get_nowait()
                        break
                    if not self.live:
                        raise RuntimeError("native worker pool interrupted; settle retained operations before resume") from self.failure
                    self.condition.wait(timeout=.1)
        reusable = True
        called = False
        try:
            called = True
            return actor(request)
        except (SemanticProjectFailure, ProjectAdmissionTimeout):
            raise
        except Exception as exc:
            if called:
                reusable = False
                with self.condition:
                    self.live -= 1
                    self.quarantined.append(actor)
                    self.failure = self.failure or exc
            raise
        finally:
            with self.condition:
                if reusable:
                    self.actors.put(actor)
                self.condition.notify_all()


def build_native_adaptive_phase1(
    *, config, native_transport, profile, guide_bytes, objective, initial_memory,
    environment, provider_transport=None, embeddings=None, project_target=None,
    max_projects=None, max_seconds=7200, request_profile=None, request_timing=None,
    profiling_guidance=None,
    reviewed_snapshot=None, trace_hook=None, trace_recorder_factory=None,
    project_workers=(), librarian_complete=None,
):
    """Compose native A3 learning without a Catlass checkout requirement.

    The caller owns discovered compiler/device settings and checked transport.
    Measurement adapters are supplied explicitly, never borrowed from A5.
    """
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_curriculum import Capability
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.phase1_adaptive_production import _knowledge_action
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.trial import _SYSTEM_PROMPT
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor

    if project_workers and reviewed_snapshot is not None:
        raise ValueError("adaptive Phase 2 remains serial")
    if project_workers:
        request_profile = project_workers[0].request_profile
        request_timing = project_workers[0].request_timing
        if any((worker.request_profile is not None, worker.request_timing is not None) !=
               (request_profile is not None, request_timing is not None) for worker in project_workers):
            raise ValueError("native workers must support the same measurement actions")

    if not isinstance(guide_bytes, bytes) or not guide_bytes:
        raise ValueError("native programming guide must be non-empty bytes")
    import hashlib
    if hashlib.sha256(guide_bytes).hexdigest() != config.guide_sha256:
        raise ValueError("native programming guide identity mismatch")
    guide_text = guide_bytes.decode("utf-8")
    profiling_mode = config.profiling_mode or ("new" if profiling_guidance is not None else "without-profiling-guidance")
    if profiling_mode not in {"new", "without-profiling-guidance"} or ((profiling_mode == "new") != (profiling_guidance is not None)):
        raise ValueError("native profiling mode and admitted guidance must agree")
    policy = a3_curriculum_policy(dtypes=config.dtypes)
    identity = {
        **ENGINE_IDENTITY,
        "schema": "ascend-native-adaptive-identity-v1", "target": config.target,
        "arch": config.arch, "execution_profile": config.execution_profile,
        "language": "ascend-c", "guide_sha256": config.guide_sha256,
        "cpl_skills_revision": config.cpl_skills_revision,
        "dtypes": config.dtypes, "model": profile.identity.as_dict(),
        "profiling_guidance": profiling_guidance,
        "profiling_mode": profiling_mode,
        "measurement_actions": {"profiling": request_profile is not None,
                                "timing": request_timing is not None},
        "kdb_collection": config.collection,
        "environment_identity": config.environment_identity or {},
    }
    if reviewed_snapshot is None:
        identity["runtime_budget"] = {"policy": "elapsed-time-v1", "max_seconds": max_seconds}
    if reviewed_snapshot is not None:
        reviewed_snapshot.verify()
        identity.update(schema="ascend-native-adaptive-phase2-identity-v1",
                        phase="adaptive-phase2", snapshot_id=reviewed_snapshot.snapshot_id)
    query, adaptive = None, None
    if config.kdb is not None:
        if embeddings is None or not config.collection:
            raise ValueError("native KDB requires explicit collection and embeddings")
        with KnowledgeDB.open_read_only(config.kdb, embeddings) as database:
            manifests = database.selected_manifests(config.collection)
            if not any(manifest.language == "ascend-c" or "ascend-c" in manifest.kernel_languages
                       for manifest in manifests):
                raise ValueError("native KDB collection must use ascend-c language")
            identity["kdb_manifest"] = (manifests[0].fingerprint if isinstance(config.collection, str)
                                        else {manifest.collection: manifest.fingerprint
                                              for manifest in manifests})
        if config.adaptive_kdb:
            from benchmarks.ascend.phase1_knowledge import NativeAdaptiveKnowledge, deepseek_librarian
            adaptive = NativeAdaptiveKnowledge(config, embeddings,
                librarian_complete or deepseek_librarian(environment))
            identity["adaptive_kdb"] = adaptive.identity
        query = adaptive.actor_query if adaptive is not None else _knowledge_action(config, embeddings)
        # The pinned local embedding model is shared; its tokenizer/model need
        # not support overlapping calls. SQLite readers/journals remain local.
        query_action, query_lock = query, Lock()
        def query(payload, context):
            if not config.adaptive_kdb or context.remaining_seconds is None:
                with query_lock:
                    return query_action(payload, context)
            import time
            deadline = time.monotonic() + context.remaining_seconds
            if not query_lock.acquire(timeout=max(0, context.remaining_seconds)):
                raise TimeoutError('actor knowledge lock elapsed allocation exhausted')
            try:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('actor knowledge lock elapsed allocation exhausted')
                return query_action(payload, replace(context, remaining_seconds=remaining))
            finally:
                query_lock.release()
    elif config.collection is not None or embeddings is not None:
        raise ValueError("KDB-off native treatment cannot receive KDB dependencies")
    curriculum_query = None
    if config.curriculum_kdb:
        from benchmarks.a5kernels.phase1_adaptive_production import curriculum_knowledge
        callback, identity["curriculum_kdb"] = curriculum_knowledge(config, embeddings)
        if config.adaptive_kdb:
            callback = adaptive.curriculum_query
        def curriculum_query(action, *, check_remaining=None):
            if check_remaining is None:
                with query_lock:
                    return callback(action)
            if not query_lock.acquire(timeout=check_remaining()):
                check_remaining()
                raise TimeoutError('curriculum query lock elapsed allocation exhausted')
            try:
                check_remaining()
                return callback(action, check_remaining=check_remaining)
            finally:
                query_lock.release()
        if config.adaptive_kdb:
            curriculum_query.begin_selection = adaptive.begin_selection
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign-identity.json"
    expected = canonical_bytes(identity)
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError("campaign root belongs to a different native target/treatment")
    else:
        if (root / "state.json").exists():
            raise ValueError("native retained state has no target identity")
        path.write_bytes(expected)
    provider = DirectCompletionProvider(environment, **(
        {"transport": provider_transport} if provider_transport is not None else {}
    ))
    actor = PiActorDriver(provider, system_prompt=_SYSTEM_PROMPT.replace("one A5 kernel", "one A3 Ascend C kernel"))
    executor = None
    if not project_workers:
        backend = NativeProjectBackend(native_transport, root / "native-evidence",
                                       target=config.target, arch=config.arch,
                                       profile=config.execution_profile)
        executor = ProjectExecutor(backend, EvidenceLedger(root / "project-evidence.jsonl"),
                                   binding=NativeProjectBinding(dtypes=config.dtypes),
                                   runtime_identity=native_target_runtime(config, native_transport))
    context = canonical_bytes({"identity": identity, "programming_guide": guide_text})
    diagnostics = "Debugger and sanitizer routes are unavailable in this native composition.\n"
    pool = None
    if project_workers:
        actors = []
        for worker in project_workers:
            worker_backend = NativeProjectBackend(worker.transport, worker.root / "native-evidence",
                target=config.target, arch=config.arch, profile=config.execution_profile)
            worker_executor = ProjectExecutor(worker_backend,
                EvidenceLedger(worker.root / "project-evidence.jsonl"),
                binding=NativeProjectBinding(dtypes=config.dtypes),
                runtime_identity=native_target_runtime(config, worker.transport))
            actors.append(AdaptiveActorProjectExecutor(worker_executor, actor, profile,
                query_knowledge=query, request_profile=worker.request_profile,
                request_timing=worker.request_timing, immutable_context=context,
                diagnostic_instructions=diagnostics, retain_semantic_failure=True,
                **({"trace_recorder_factory": trace_recorder_factory}
                   if trace_recorder_factory is not None else {})))
        pool = NativeProjectPool(actors)
    _, coordinator = compose_adaptive_controller(
        root=root, objective=objective, initial_memory=initial_memory,
        executor=executor, actor=actor, profile=profile, provider=provider,
        immutable_context=context,
        project_actor=pool, max_parallel=len(project_workers) or 1,
        target_policy=policy, profiling_guidance=profiling_guidance,
        trace_recorder_factory=trace_recorder_factory,
        curriculum_query_knowledge=curriculum_query,
        query_knowledge=query, request_profile=request_profile, request_timing=request_timing,
        diagnostic_instructions=diagnostics,
        project_target=project_target, max_projects=max_projects, max_seconds=max_seconds,
        reviewed_snapshot=reviewed_snapshot, trace_hook=trace_hook,
        unavailable_capabilities=tuple(capability for capability, action in (
            (Capability.PROFILING, request_profile), (Capability.TIMING_SWEEP, request_timing),
        ) if action is None),
    )
    if reviewed_snapshot is None:
        from benchmarks.ascend.wave_environment import NativeWaveEnvironmentCheck
        coordinator.environment_check = NativeWaveEnvironmentCheck(config,
            tuple(worker.transport for worker in project_workers) or (native_transport,))
    if adaptive is not None:
        def reference_query(action, *, check_remaining, **context):
            if not query_lock.acquire(timeout=max(0, check_remaining())):
                raise TimeoutError('reference query lock elapsed allocation exhausted')
            try:
                check_remaining()
                return adaptive.reference_query(action, check_remaining=check_remaining, **context)
            finally:
                query_lock.release()
        coordinator.contextual_reference_query = reference_query
    return coordinator


@dataclass(frozen=True)
class NativeKernelABI(ProjectKernelABI):
    @property
    def signature(self):
        args = ", ".join(f"GM_ADDR {name}" for name in (*self.input_arguments, self.output_argument))
        return f'extern "C" __global__ __aicore__ void {self.kernel_name}({args})'


class NativeProjectBinding:
    def __init__(self, *, dtypes=("float32",)):
        self.policy = a3_curriculum_policy(dtypes=dtypes)

    def family(self, spec):
        operations = {node.op for node in spec.graph.nodes}
        if ProjectOp.MATMUL in operations:
            return ProjectExecutionFamily.CUBE_MATMUL
        if operations & {ProjectOp.REDUCE_SUM, ProjectOp.MASK_POSITIVE}:
            return ProjectExecutionFamily.VECTOR_REDUCTION_MASK
        return ProjectExecutionFamily.MOVEMENT_ELEMENTWISE

    def abi(self, spec, inputs):
        self.policy.admit(spec)
        return NativeKernelABI("project_kernel", tuple(f"gm_input{i}" for i in range(len(inputs))), "gm_output")

    def instruction(self, spec, family, abi, inputs_sha256):
        contract = {
            "project": spec.as_dict(), "kernel_abi": abi.as_dict(),
            "inputs_sha256": inputs_sha256, "execution_family": family.value,
        }
        return (
            f"Implement the complete Ascend C device source defining {abi.signature}. "
            "Include kernel_operator.h; host code owns compilation and launching. "
            "Arguments are contiguous physical padded tensor storage in graph input order "
            "then output. Shapes and dtype are fixed by the project contract, not runtime "
            "parameters. Host macros RSI_NATIVE_INPUT<i>_NUMEL, RSI_NATIVE_INPUT<i>_DIM<axis>, "
            "RSI_NATIVE_OUTPUT_NUMEL and RSI_NATIVE_OUTPUT_DIM<axis> describe physical extents. "
            "The corresponding *_LOGICAL_NUMEL and *_LOGICAL_DIM<axis> macros describe "
            "logical extents, excluding sentinel padding; use them for logical arithmetic. "
            "For timing sweeps use these macros rather than hardcoded lengths: the host "
            "recompiles the unchanged source for each requested shape. Round the last dimension "
            "up to padding_multiple to obtain physical "
            "storage. Compute only logical elements; input padding contains nonzero sentinels. "
            "Write every physical output element, setting padding to zero. Use the declared "
            "block_count for disjoint work partitioning. Do not register Torch operators, "
            "provide host code, or substitute a framework operation for the kernel. "
            "Compile-only observations are not correctness evidence; inspect host results. "
            "Project contract: " + canonical_bytes(contract).decode("ascii")
        )


class NativeProjectBackend:
    """Adapt checked native compilation/raw receipts to the shared host oracle."""

    def __init__(self, transport, root, *, target, arch, profile):
        self.transport, self.root = transport, Path(root)
        self.target, self.arch, self.profile = target, arch, profile
        self._compilations = {}

    def plan(self, request):
        identity = canonical_digest({"source": request.candidate_source,
                                     "spec": request.spec.as_dict(),
                                     "inputs": request.inputs_sha256})
        return NativeCandidatePlan(
            request.candidate_source, request.abi.kernel_name,
            tuple(NativeTensor(name, tensor.physical_shape, request.spec.dtype.value,
                               tensor.storage, logical_shape=tensor.shape) for name, tensor in
                  zip(request.abi.input_arguments, request.inputs)),
            NativeTensor(request.abi.output_argument, request.expected.physical_shape,
                         request.expected.dtype, logical_shape=request.expected.shape),
            identity, identity, request.spec.project_id, self.profile,
            self.target, self.arch, request.spec.block_count,
        )

    def _retain(self, payload):
        digest = canonical_digest(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{digest}.json").write_bytes(canonical_bytes(payload))
        return digest

    def compile(self, request):
        try:
            plan = self.plan(request)
        except ValueError as exc:
            digest = self._retain({"stage": "source-admission", "diagnostic": str(exc)})
            return CompileObservation(False, digest, str(exc), None)
        try:
            result = self.transport.compile(plan, self.root / plan.execution_id)
        except (_PendingObservation, OSError) as exc:
            raise ProjectInfrastructureError(str(exc)) from exc
        if isinstance(result, FailedEvidence):
            if native_infrastructure_failure(result):
                raise ProjectInfrastructureError(result.detail)
            details = result.attestation_payload()
            digest = self._retain(details)
            return CompileObservation(False, digest, result.detail, None, details)
        self._compilations[plan.execution_id] = result
        details = {"stage": "compile", "stdout": result.stdout,
                   "stderr": result.stderr, "library_sha256": result.library_sha256}
        digest = self._retain(details)
        return CompileObservation(True, digest, result.stdout or "native compilation completed", plan.execution_id, details)

    def run(self, request, compilation):
        plan = self.plan(request)
        if plan.execution_id != compilation.artifact_id:
            raise ValueError("native runtime request differs from compiled candidate")
        try:
            receipt = self.transport.execute_receipt(self._compilations[compilation.artifact_id])
        except (_PendingObservation, OSError) as exc:
            raise ProjectInfrastructureError(str(exc)) from exc
        if isinstance(receipt, FailedEvidence):
            if native_infrastructure_failure(receipt):
                raise ProjectInfrastructureError(receipt.detail)
            digest = self._retain(receipt.attestation_payload())
            return RuntimeObservation(False, digest, receipt.detail, None)
        payload = {"exit_code": receipt.exit_code, "stdout": receipt.stdout,
                   "stderr": receipt.stderr, "output": list(receipt.output),
                   "handle": receipt.job_handle, "metadata": list(receipt.metadata)}
        digest = self._retain(payload)
        passed = receipt.exit_code == 0
        return RuntimeObservation(passed, digest, receipt.stdout or receipt.stderr or "native launch completed",
                                  receipt.output if passed else None)
