"""Shared Phase-1 launcher: select a target once, preserve existing A5 CLI."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from benchmarks.a3kernels.phase1_protocol import FailedEvidence
from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend
from benchmarks.a3kernels.native_measurements import NativeMeasurementBackend
from benchmarks.a3kernels.native_candidate import native_infrastructure_failure
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.ascend.production import (
    NativeProductionConfig, NativeProjectBackend, NativeProjectWorker, build_native_adaptive_phase1,
)
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_adaptive_production import _measurement_action
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile
from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
from benchmarks.a5kernels.phase1_project_measurements import specialize_timing
from benchmarks.a5kernels.phase1_production import load_direct_environment
from benchmarks.a5kernels.evidence import canonical_digest


class NativeProjectMeasurements:
    """Bind shared prepared projects and measurement actions to native replay."""

    def __init__(self, backend, config, root, *, factory=NativeMeasurementBackend):
        self.backend, self.config, self.root, self.factory = backend, config, Path(root), factory

    def _measurement(self, request):
        plan = self.backend.plan(request)
        try:
            compiled = self.backend.transport.compile(plan, self.root / plan.execution_id)
        except (_PendingObservation, OSError) as exc:
            raise ProjectInfrastructureError(str(exc)) from exc
        if isinstance(compiled, FailedEvidence):
            if native_infrastructure_failure(compiled):
                raise ProjectInfrastructureError(compiled.detail)
            raise ValueError("measurement compilation failed: " + compiled.detail)
        tolerance = {"float32": 1e-5, "float16": 1e-2, "bfloat16": 2e-2}[request.expected.dtype]
        return self.factory(
            compilation=compiled, candidate_backend=self.backend.transport,
            expected_output=request.expected.storage, atol=tolerance, rtol=tolerance,
            environment_identity=self.config["environment_identity"],
            validation_wrapper=self.config["validation_wrapper"], cpl_remote=self.config["cpl_remote"],
            profile=self.config["execution_profile"],
            remote_candidate_directory=str(self.backend.transport.remote_candidate_directory(plan)),
            evidence_directory=self.root / plan.execution_id / "measurements",
            physical_device=self.config["physical_device"],
            supported_metrics=self.config.get("supported_metrics", ["Basic", "PipeUtilization"]),
        )

    @staticmethod
    def _combine(results, summary, source):
        artifacts = {}
        candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        for result in results:
            artifacts.update(result["artifact_payloads"])
            payload = {"identity": result["identity"], "summary": result["summary"],
                       "candidate_sha256": candidate_sha256}
            artifacts[canonical_digest(payload)] = payload
        return dict(summary=summary, artifact_payloads=artifacts, evidence_sha256=tuple(artifacts))

    def timing(self, request):
        results = []
        for spec in specialize_timing(request.spec):
            prepared_inputs = materialize_inputs(spec)
            prepared = replace(request, spec=spec, inputs=prepared_inputs,
                expected=evaluate_reference(spec, prepared_inputs),
                inputs_sha256=canonical_digest({"tensors": [dict(shape=list(t.shape),
                    physical_shape=list(t.physical_shape), storage=list(t.storage)) for t in prepared_inputs]}))
            try:
                results.append(self._measurement(prepared).timing(warm_up=5, launch_count=20))
            except (_PendingObservation, OSError) as exc:
                raise ProjectInfrastructureError(str(exc)) from exc
        return self._combine(results, {"timing": [
            {"shape": r["identity"]["output"]["logical_shape"],
             "physical_shape": r["identity"]["output"]["shape"], **r["summary"]} for r in results]},
             request.candidate_source)

    def profile(self, request, metric):
        if metric != "pipe-utilization":
            raise ValueError("native profiling metric unavailable; use pipe-utilization")
        try:
            result = self._measurement(request).capture("PipeUtilization")
        except (_PendingObservation, OSError) as exc:
            raise ProjectInfrastructureError(str(exc)) from exc
        return self._combine([result], result["summary"], request.candidate_source)


def _guidance(path):
    root = Path(path)
    text = "\n\n".join((root / name).read_text() for name in (
        "SKILL.md", "references/a3-metric-semantics.md"))
    return ("RSI A3 actions: after HOST_VERIFIED, request-timing:{} measures an ordinary "
            "timing sweep; request-profile:{} captures the project's pipe-utilization. "
            "No memory-throughput/timeline action is available. These are harness "
            "actions, not shell commands. Preserve returned evidence identities.\n\n" + text)


def load_native_config(path):
    """Load machine-local settings; persist content identities, never this mapping."""
    config = json.loads(Path(path).read_text())
    required = ("guide", "target", "arch", "execution_profile", "cpl_skills_revision",
                "cpl_remote", "validation_wrapper", "remote_workspace", "environment_identity")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("native runtime config missing fields: " + ", ".join(missing))
    if any(not config["environment_identity"].get(k) for k in ("cann_version", "profiler_version")):
        raise ValueError("native runtime config requires discovered CANN/profiler versions")
    devices = native_devices(config)
    # Phase 2 consumes the same runtime file but deliberately remains serial.
    config.setdefault("physical_device", devices[0])
    return config


def native_devices(config):
    """Explicit, caller-checked device pool; omission preserves serial behavior."""
    devices = config.get("physical_devices", [config.get("physical_device")])
    if (not isinstance(devices, list) or not devices
            or any(type(device) is not int or device < 0 for device in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("physical_devices must be a nonempty list of distinct nonnegative device IDs")
    return devices


def native_transport(config, root, *, factory=BZA3RemoteCandidateBackend):
    return factory(cpl_remote=config["cpl_remote"], validation_wrapper=config["validation_wrapper"],
        profile=config["execution_profile"], remote_workspace=config["remote_workspace"],
        physical_device=config["physical_device"], state_directory=Path(root) / "transport-state")


def main(argv=None, *, a5_main=None, native_builder=build_native_adaptive_phase1,
         transport_factory=BZA3RemoteCandidateBackend, provider_transport=None,
         environment_loader=load_direct_environment, embedding_factory=None,
         measurement_factory=NativeMeasurementBackend, native_parallel=True):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--target", required=True, choices=("a5", "a3"))
    target, rest = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    if target.target == "a5":
        if a5_main is None:
            from benchmarks.a5kernels.phase1_adaptive_cli import main as a5_main
        return a5_main(rest)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--initial-memory", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", choices=[i.backend_model.value for i in MODEL_IDENTITIES])
    parser.add_argument("--knowledge", choices=("with-kdb", "without-kdb"))
    parser.add_argument("--profiling", choices=("new", "without-profiling-guidance"))
    parser.add_argument("--project-target", type=int)
    parser.add_argument("--max-projects", type=int)
    args = parser.parse_args(rest)
    memory = args.initial_memory.read_bytes()
    if args.action == "report":
        identity = json.loads((args.root / "campaign-identity.json").read_text())
        if identity.get("schema") != "ascend-native-adaptive-identity-v1" or identity.get("language") != "ascend-c":
            raise ValueError("--target a3 requires native A3 retained campaign identity")
        result = AdaptivePhase1Coordinator.read_retained_report(args.root, args.objective, memory)
    else:
        if any(v is None for v in (args.config, args.model, args.knowledge, args.profiling)):
            parser.error("live native actions require --config, --model, --knowledge, --profiling")
        config = load_native_config(args.config)
        guide = Path(config["guide"]).read_bytes()
        profile = direct_profile(next(i for i in MODEL_IDENTITIES if i.backend_model.value == args.model))
        embeddings, kdb, collection = None, None, None
        if args.knowledge == "with-kdb":
            from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
            from benchmarks.a5kernels.production import _transformers_cache
            factory = embedding_factory or PinnedBGEEmbeddings
            embeddings = factory(cache_dir=_transformers_cache(Path(config["bge_cache"])))
            kdb, collection = Path(config["kdb"]), config["collection"]
        native_config = NativeProductionConfig(
            args.root, config["target"], config["arch"], config["execution_profile"],
            hashlib.sha256(guide).hexdigest(), config["cpl_skills_revision"],
            tuple(config.get("dtypes", ["float32"])), kdb, collection,
            environment_identity=config["environment_identity"],
            profiling_mode=args.profiling,
        )
        workers = []
        parallel = native_parallel and "physical_devices" in config
        for device in native_devices(config) if parallel else [config["physical_device"]]:
            worker_root = (args.root / "native-workers" / f"device-{device}"
                           if parallel else args.root)
            worker_config = {**config, "physical_device": device}
            transport = native_transport(worker_config, worker_root, factory=transport_factory)
            backend = NativeProjectBackend(transport, worker_root / "native-measurement-evidence",
                target=config["target"], arch=config["arch"], profile=config["execution_profile"])
            measurements = NativeProjectMeasurements(backend, worker_config, worker_root / "measurements",
                                                      factory=measurement_factory)
            workers.append(NativeProjectWorker(worker_root, transport,
                _measurement_action(measurements, profiling=True),
                _measurement_action(measurements, profiling=False)))
        first = workers[0]
        guidance = _guidance(config["profiling_guide"]) if args.profiling == "new" else None
        result = getattr(native_builder(
            config=native_config, native_transport=first.transport, profile=profile, guide_bytes=guide,
            objective=args.objective, initial_memory=memory,
            environment=environment_loader(env_file=args.env_file), provider_transport=provider_transport,
            embeddings=embeddings, project_target=args.project_target, max_projects=args.max_projects,
            request_timing=first.request_timing, request_profile=first.request_profile,
            **({"project_workers": tuple(workers)} if parallel else {}),
            profiling_guidance=guidance,
        ), args.action)()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
