"""Explicit phase campaigns, independent artifact collection, portable checkpoints."""
import argparse
from contextlib import redirect_stdout
from functools import wraps
import fcntl
import hashlib
import io
import math
import json
from benchmarks.ascend.status_descriptions import describe_statuses
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from .artifact_collector import ArtifactCollector
from .training_annex import inventory_campaign, package_campaign


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def _harness_revision(root=None):
    """Identify a checkout or git-exported source without inventing provenance."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    revision = ""
    if (root / ".git").exists():
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        archival = root / ".git_archival.txt"
        revision = archival.read_text().strip().removeprefix("node: ") if archival.is_file() else ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("harness revision unavailable: use a Git checkout or git archive with export-subst metadata")
    return revision


def _snapshot(job, options):
    """Read the frozen copy on resume, never recopy a growing Phase 1."""
    from benchmarks.a5kernels.phase2_adaptive_memory import (
        AdaptivePhase1Snapshot, adaptive_destination, admit_adaptive_snapshot,
    )
    from .phase2 import NativePhase1Snapshot, NativePhase2Cell, admit_native_snapshot
    control = Path(job["control"])
    retained = control / "snapshot-location.json"
    if retained.exists():
        root = Path(_read(retained)["path"])
        manifest = _read(root / "snapshot.json")
        if job["target"] == "a3":
            identity = manifest["destination_identity"]
            cell = NativePhase2Cell(identity["source_identity"], identity["qualification_sha256"])
            snapshot = NativePhase1Snapshot(root, cell, manifest)
        else:
            cell = adaptive_destination(options["cell"], _read(root / "review.json"))
            snapshot = AdaptivePhase1Snapshot(root, cell, manifest)
        snapshot.verify()
        return snapshot
    memory = (Path(job["root"]) / "initial-memory.bin").read_bytes()
    if job["target"] == "a3":
        snapshot = admit_native_snapshot(job["phase1_root"], memory, job["review"], job["root"])
    else:
        snapshot = admit_adaptive_snapshot(job["phase1_root"], memory, job["review"],
                                           options["cell"], job["root"])
    _write(retained, {"path": str(snapshot.root)})
    return snapshot


def _target_available_tools(actor):
    """Advertise composed routes, never tools invented by campaign settings."""
    from benchmarks.a5kernels.phase1_adaptive_actor import (
        disabled_profile_action, disabled_diagnostic_action,
    )
    tools = list(getattr(getattr(actor, "executor", None), "supported_tools", ()))
    for attribute, disabled, names in (
        ("request_profile", disabled_profile_action, ("request-profile",)),
        ("request_timing", disabled_profile_action, ("request-timing",)),
        ("request_diagnostic", disabled_diagnostic_action, ("request-debug", "request-sanitizer")),
    ):
        route = getattr(actor, attribute, None)
        if callable(route) and route is not disabled:
            tools.extend(names)
    return tuple(tools)


def _composed_target_runtime(actor, declared):
    actual = getattr(getattr(actor, "executor", None), "runtime_identity", None)
    if actual is None:
        raise ValueError("target goals require observed compiler/runtime identity on the composed executor")
    if dict(actual) != declared:
        raise ValueError("qualified target runtime differs from the composed executor")
    return dict(actual)


def build_controller(job):
    """Reuse the existing native/Catlass CLI bindings, replacing only execution."""
    from .phase1_cli import main as target_main
    from .production import build_native_adaptive_phase1
    from benchmarks.a5kernels.phase1_adaptive_cli import main as a5_main
    from benchmarks.a5kernels.phase1_adaptive_production import build_adaptive_phase1
    from .training_traces import recorder_factory

    config_bytes = Path(job["config"]).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != job["config_sha256"]:
        raise ValueError("runtime settings changed; use a new campaign configuration")
    settings = json.loads(config_bytes)
    if job['phase'] == 1 and settings.get('automatic_remediation') and settings.get('optimizer_worker_config'):
        worker_bytes = Path(settings['optimizer_worker_config']).read_bytes()
        if hashlib.sha256(worker_bytes).hexdigest() != job.get('optimizer_worker_config_sha256'):
            raise ValueError('optimizer worker configuration changed on resume')
    captured = []

    def capture(**options):
        model = (options["profile"].identity if job["target"] == "a3" else options["cell"].model)
        if model.backend_model.value != job["model"]:
            raise ValueError("configured model label differs from selected provider")
        if job["phase"] == 2:
            snapshot = _snapshot(job, options)
            if job["target"] == "a3":
                source = snapshot.destination.source_identity
                native = options["config"]
                for key in ("target", "arch", "execution_profile", "guide_sha256",
                            "cpl_skills_revision", "environment_identity"):
                    if getattr(native, key) != source[key]:
                        raise ValueError("runtime differs from reviewed Phase 1: " + key)
                if native.collection != source.get("kdb_collection"):
                    raise ValueError("KDB collections differ from reviewed Phase 1")
                if options["profile"].identity.as_dict() != source["model"]:
                    raise ValueError("model differs from reviewed Phase 1")
                if native.profiling_mode != source.get("profiling_mode"):
                    raise ValueError("profiling treatment differs from reviewed Phase 1")
            options["reviewed_snapshot"] = snapshot
        options["trace_recorder_factory"] = recorder_factory(
            phase=f"phase{job['phase']}", round_id=job["round_id"],
            model=job["model"], source_manifest_sha256=job["source_manifest_sha256"],
        )
        # Campaign settings opt into record-only verification without changing CLI defaults.
        if settings.get("verify_memory_usefulness"):
            options["usefulness_enabled"] = True
            options["usefulness_pair_seconds"] = settings.get("usefulness_pair_seconds", 600)
        built = (build_native_adaptive_phase1 if job["target"] == "a3"
                 else build_adaptive_phase1)(**options)
        if job["phase"] == 2 and job["target"] == "a3":
            identity_path = Path(job["root"]) / "campaign-identity.json"
            if identity_path.exists():
                current = _read(identity_path)
                for key, value in snapshot.destination.source_identity.items():
                    if key not in {"schema", "runtime_budget"} and current.get(key) != value:
                        raise ValueError("native treatment changed reviewed Phase 1: " + key)
        coordinator = getattr(built, "coordinator", built)
        if job.get('feedback_root') and job['phase'] == 1:
            from .failure_curriculum import FailureCurriculum
            data = (Path(job['root']) / 'phase2-feedback.json').read_bytes()
            if hashlib.sha256(data).hexdigest() != job['feedback_selection_sha256']:
                raise ValueError('frozen Phase 2 feedback changed')
            feedback = FailureCurriculum(job['root'], job['feedback_root'], json.loads(data))
            coordinator.failure_feedback = feedback
            coordinator.curriculum.failure_feedback = feedback
        if job['phase'] == 1 and settings.get('automatic_remediation', False):
            from .automatic_remediation import AutomaticRemediation, optimizer_retry_adapter
            from optimizer.worker import load_config
            automatic = AutomaticRemediation(coordinator, target=job['target'],
                store=settings.get('feedback_root', job.get('feedback_root')),
                sources=settings.get('feedback_sources', []),
                seconds=settings.get('failure_active_seconds', 1800))
            if settings.get('optimizer_worker_config'):
                automatic.retry_external = optimizer_retry_adapter(coordinator, automatic.store,
                    load_config(settings['optimizer_worker_config']))
        if job.get('phase1_original_retry'):
            from .manual_retry import ManualPhase1Retry
            data = (Path(job['root']) / 'phase2-feedback.json').read_bytes()
            if hashlib.sha256(data).hexdigest() != job['feedback_selection_sha256']:
                raise ValueError('Frozen original retry selection changed')
            coordinator = ManualPhase1Retry(coordinator, job, json.loads(data))
        elif job.get("optimizer_goal"):
            coordinator = _optimizer_controller(job, settings, snapshot, coordinator, options["config"])
        elif job.get("target_goal") is not None or job.get("autonomous_goal"):
            from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
            from .phase2_targets import TargetContract
            from .phase2_target_selector import TargetCurriculumAgent
            from .phase2_target_loop import TargetDrivenPhase2Coordinator
            if job.get("autonomous_goal"):
                from .phase2_goal_bootstrap import bootstrap_goal
                from .phase2_time import Phase2TimeBudget
                budget = Phase2TimeBudget(job["root"], job["phase2_max_seconds"],
                                          job["practice_max_seconds"])
                with budget.active():
                    goal, artifacts = bootstrap_goal(coordinator, root=job["root"],
                        private_root=Path(job["control"]) / "bootstrap", objective=job["objective"],
                        available_tools=_target_available_tools(coordinator.execute_project),
                        remaining_seconds=lambda: budget.remaining)
                runtime = goal.runtime
            else:
                goal = TargetContract.from_mapping(job["target_goal"])
                artifacts = ContentAddressedArtifacts(Path(job["target_artifact_store"]))
                runtime = _composed_target_runtime(coordinator.execute_project, settings["target_runtime"])
            usefulness_verifier = getattr(coordinator, "usefulness_verifier", None)
            coordinator = TargetDrivenPhase2Coordinator(
                job["root"], job["objective"], snapshot,
                TargetCurriculumAgent(coordinator.curriculum._complete,
                                      target_policy=coordinator.curriculum.target_policy),
                coordinator.execute_project,
                target=goal, artifacts=artifacts, runtime=runtime,
                target_policy=coordinator.curriculum.target_policy,
                max_seconds=job["phase2_max_seconds"],
                max_practice_seconds=job["practice_max_seconds"],
                available_tools=_target_available_tools(coordinator.execute_project),
                trace_hook=coordinator.trace_hook,
            )
            if usefulness_verifier is not None:
                coordinator.usefulness_verifier = usefulness_verifier
                usefulness_verifier.remaining_seconds = coordinator._remaining_seconds
        captured.append(coordinator)
        # The old CLI calls run() after composing. Capture without executing it.
        return type("Deferred", (), {"run": lambda self: {"deferred": True}})()

    initial_memory = job.get("initial_memory", str(Path(job["root"]) / "initial-memory.bin"))
    arguments = ["run", "--root", job["root"], "--objective", job["objective"],
                 "--initial-memory", initial_memory,
                 "--project-target", str(job["project_target"]), *settings["phase1_arguments"]]
    if job.get("worker_config"):
        data = Path(job["worker_config"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != job["worker_config_sha256"]:
            raise ValueError("worker execution configuration changed on resume")
        if "--worker-config" in arguments:
            raise ValueError("choose campaign --worker-config or phase1_arguments worker config")
        arguments += ["--worker-config", job["worker_config"]]
    with redirect_stdout(io.StringIO()):
        target_main(["--target", job["target"], *arguments], native_builder=capture,
                    native_parallel=job["phase"] == 1,
                    a5_main=lambda argv: a5_main(argv, builder=capture))
    return captured[0]


def _optimizer_controller(job, settings, snapshot, composed, native):
    """Bind the merged optimizer worker to the reviewed native campaign lineage."""
    from optimizer.memory import freeze_reviewed_phase1
    from optimizer.worker import load_config, knowledge_query
    from .phase2_optimizer_execution import OptimizerTaskExecutor
    from .phase2_optimizer_campaign import OptimizerPhase2Coordinator
    from .phase2_optimizer_continuation import campaign_seed

    path = settings.get("optimizer_worker_config")
    if not path:
        raise ValueError("optimizer Phase 2 requires optimizer_worker_config in ignored campaign settings")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != job["optimizer_worker_config_sha256"]:
        raise ValueError("optimizer worker configuration changed on resume")
    config = load_config(path, job.get("worker_config"))
    runtime = getattr(getattr(composed.execute_project, "executor", None), "runtime_identity", None)
    if (("local" if config.get("execution") == "local" else config["profile"]) != native.execution_profile or config["arch"] != native.arch
            or runtime is None or str(config["physical_device"]) != str(runtime["device"])
            or hashlib.sha256(Path(config["guide"]).read_bytes()).hexdigest() != native.guide_sha256
            or config.get("model_profile", "deepseek-flash") != job["model"]):
        raise ValueError("optimizer runtime, device, guide or model differs from reviewed composition")
    selected = config.get("knowledge")
    if bool(selected) != bool(native.collection) or (selected and selected.get("collection") != native.collection):
        raise ValueError("optimizer KDB treatment differs from reviewed composition")
    if selected:
        reviewed = snapshot.destination.source_identity
        if ("manifest" not in selected or "kdb_manifest" not in reviewed
                or selected["manifest"] != reviewed["kdb_manifest"]):
            raise ValueError("optimizer KDB manifest differs from reviewed Phase 1")
    query = knowledge_query(config, Path(job["root"]) / "curriculum")
    seed = campaign_seed(job["root"], freeze_reviewed_phase1(snapshot),
                         settings.get("optimizer_prior_campaigns", []))
    retry = job.get('retry_spec')
    if retry and config.get('max_cases', 24) != retry['max_cases']:
        raise ValueError('retry max_cases differs from original frozen suite')
    def executor(directory, task, memory):
        return OptimizerTaskExecutor(directory, task, memory, config,
            expected_baseline=retry if retry and task == retry['task'] else None)
    return OptimizerPhase2Coordinator(job["root"], job["objective"], seed,
        composed.curriculum._complete,
        executor,
        query_knowledge=(lambda text: {"results": query(text)}) if query else None,
        max_seconds=job["phase2_max_seconds"],
        max_practice_seconds=job["practice_max_seconds"],
        feedback_root=job.get('feedback_root', settings.get("phase2_feedback_root")),
        fixed_task=retry['task'] if retry else None,
        usefulness_verifier=getattr(composed, 'usefulness_verifier', None))


def _approval_path(job):
    return Path(job["control"]) / "curriculum-approval.json"


def _argument_value(arguments, option):
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
    return None


def _trace_provenance(root):
    records = []
    for path in Path(root).rglob("*.jsonl"):
        if path.parent.name not in {"sft-traces", "traces"}:
            continue
        try:
            first = next(line for line in path.read_text().splitlines() if line.strip())
            record = json.loads(first)
            if record.get("kind") == "header":
                records.append(record.get("payload", {}))
        except (OSError, ValueError, StopIteration):
            raise ValueError(f"invalid source provenance trace: {path}") from None
    return records


def _process_option(argv, option, cwd):
    for index, token in enumerate(argv):
        value = (argv[index + 1] if token == option and index + 1 < len(argv) else
                 token.split("=", 1)[1] if token.startswith(option + "=") else None)
        if value is not None:
            path = Path(value)
            return (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    return None


def _active_campaign_processes(root, *, proc_root=Path("/proc"), current_pid=None):
    """Return bounded Linux process evidence for Python workers naming this root."""
    root = str(Path(root).resolve())
    current_pid = os.getpid() if current_pid is None else current_pid
    matches = []
    try:
        entries = sorted((path for path in Path(proc_root).iterdir()
                          if path.name.isdigit()), key=lambda path: int(path.name))[:4096]
    except OSError:
        return matches
    for process in entries:
        pid = int(process.name)
        if pid == current_pid:
            continue
        try:
            with (process / "cmdline").open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                continue
            argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
            cwd = (process / "cwd").resolve(strict=True)
        except OSError:
            continue
        named_root = _process_option(argv, "--root", cwd)
        names_root = named_root is not None and str(named_root) == root
        control = _process_option(argv, "--control", cwd)
        if not names_root and "_campaign" in argv and control is not None:
            try:
                job = _read(control / "job.json")
                names_root = (job.get("schema") == "ascend-campaign-run-v1"
                              and str(Path(job["root"]).resolve()) == root)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        python_worker = any("python" in Path(token).name.lower() or token.endswith(".py")
                            for token in argv[:3])
        if names_root and python_worker:
            matches.append({"pid": pid, "command": " ".join(argv[:4])[:300]})
    return matches


def adopt_campaign(root, control, objective, initial_memory, config, legacy_revision, *,
                   model=None, source_manifest=None, knowledge=None, profiling=None,
                   env_file=None):
    """Adopt a completed native Phase 1 boundary without rewriting its history."""
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.sft_dataset import SourceManifest

    root = Path(root).resolve(strict=True)
    control = Path(control).resolve()
    initial_memory = Path(initial_memory).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    if control.exists():
        raise FileExistsError(f"control directory already exists: {control}")
    if root == control or root in control.parents or control in root.parents:
        raise ValueError("campaign and control directories must be separate")
    if not re.fullmatch(r"[0-9a-f]{40}", legacy_revision):
        raise ValueError("legacy harness revision must be an exact Git revision")
    active = _active_campaign_processes(root)
    if active:
        raise ValueError("active legacy campaign process still references the campaign root")
    if (root / "pending.json").exists():
        raise ValueError("legacy campaign still has a pending wave")
    identity = _read(root / "campaign-identity.json")
    if (identity.get("schema") != "ascend-native-adaptive-identity-v1"
            or identity.get("language") != "ascend-c"):
        raise ValueError("legacy campaign has incompatible A3 Phase 1 identity")
    state = _read(root / "state.json")
    retained_state = (root / "state.json").read_bytes()
    retained_identity = (root / "campaign-identity.json").read_bytes()
    if state.get("terminal") is not None:
        raise ValueError("legacy campaign must be at TARGET_REACHED without a terminal decision")
    seed = initial_memory.read_bytes()
    report = AdaptivePhase1Coordinator.read_retained_report(root, objective, seed)
    if report["status"] != "TARGET_REACHED":
        raise ValueError("legacy campaign must be at TARGET_REACHED")

    supplied = json.loads(config.read_bytes())
    # Retain compatibility for already-created test/control records while the
    # public adopt CLI accepts the native config directly.
    if "phase1_arguments" in supplied:
        arguments = supplied["phase1_arguments"]
        model = model or supplied.get("model")
        source_manifest = source_manifest or supplied.get("source_manifest")
        knowledge = knowledge or _argument_value(arguments, "--knowledge")
        profiling = profiling or _argument_value(arguments, "--profiling")
        native_path = _argument_value(arguments, "--config")
    else:
        if None in (model, source_manifest, knowledge, profiling):
            raise ValueError("direct native adoption requires model, source manifest, knowledge and profiling")
        native_path = str(config)
        arguments = ["--config", native_path, "--model", model,
                     "--knowledge", knowledge, "--profiling", profiling]
        if env_file is not None:
            arguments += ["--env-file", str(Path(env_file).resolve(strict=True))]
    if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
        raise ValueError("runtime phase1_arguments must be a string list")
    if (_argument_value(arguments, "--model") != model
            or identity.get("model", {}).get("backend_model") != model):
        raise ValueError("model provenance differs from retained A3 identity")
    if native_path is None:
        raise ValueError("runtime configuration has no native --config")
    native_path = Path(native_path).resolve(strict=True)
    from .phase1_cli import load_native_config
    native = load_native_config(native_path)
    guide = Path(native["guide"]).resolve(strict=True).read_bytes()
    from .phase1_cli import _guidance
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile
    expected_identity = {
        "schema": "ascend-native-adaptive-identity-v1",
        "target": native.get("target"), "arch": native.get("arch"),
        "execution_profile": native.get("execution_profile"),
        "language": "ascend-c",
        "guide_sha256": hashlib.sha256(guide).hexdigest(),
        "cpl_skills_revision": native.get("cpl_skills_revision"),
        "dtypes": native.get("dtypes", ["float32"]),
        "model": direct_profile(next(item for item in MODEL_IDENTITIES
                                      if item.backend_model.value == model)).identity.as_dict(),
        "profiling_guidance": (_guidance(native["profiling_guide"])
                               if profiling == "new" else None),
        "environment_identity": native.get("environment_identity"),
        "profiling_mode": profiling,
        "measurement_actions": {"profiling": True, "timing": True},
        "kdb_collection": native.get("collection") if knowledge == "with-kdb" else None,
    }
    if knowledge == "with-kdb":
        import sqlite3
        from benchmarks.a5kernels.knowledge import CollectionManifest, collection_names
        connection = sqlite3.connect(Path(native["kdb"]).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            manifests = [CollectionManifest.from_json(connection.execute(
                "SELECT manifest_json FROM collections WHERE name=?", (name,)).fetchone()[0])
                for name in collection_names(native["collection"])]
        finally:
            connection.close()
        expected_identity["kdb_manifest"] = (manifests[0].fingerprint
            if isinstance(native["collection"], str)
            else {manifest.collection: manifest.fingerprint for manifest in manifests})
    if identity != expected_identity:
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise ValueError(f"runtime configuration differs from retained identity: {key}")
        unexpected = sorted(set(identity) - set(expected_identity))
        if unexpected:
            raise ValueError("retained identity has unsupported behavior fields: " + ", ".join(unexpected))
    source_path = Path(source_manifest).resolve(strict=True)
    source = SourceManifest.load(source_path)
    traces = _trace_provenance(root)
    historical_digests = {row.get("source_manifest_sha256") for row in traces}
    if (any(row.get("phase") != "phase1" or row.get("model") != model for row in traces)
            or any(not isinstance(digest, str)
                   or not re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in historical_digests)
            or len(historical_digests) > 1):
        raise ValueError("source provenance in retained traces is invalid or inconsistent")
    historical_digest = next(iter(historical_digests), None)
    historical_source = "validated" if traces else "unavailable"
    source_scope = ("historical-and-post-adoption"
                    if historical_digest == source.sha256 else "post-adoption")

    adopting_revision = _harness_revision()
    created = time.time()
    wrapper = {"model": model, "source_manifest": str(source_path),
               "phase1_arguments": arguments}
    wrapper_bytes = json.dumps(wrapper, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    job = {"schema": "ascend-campaign-run-v1", "target": "a3", "phase": 1,
           "objective": objective, "project_target": report["project_target"],
           "model": model, "source_manifest_sha256": source.sha256,
           "harness_revision": adopting_revision,
           "legacy_harness_revision": legacy_revision,
           "adopting_harness_revision": adopting_revision,
           "historical_code_provenance": legacy_revision,
           "historical_source_provenance": historical_source,
           "historical_source_manifest_sha256": historical_digest,
           "source_provenance_scope": source_scope,
           "source_provenance_activation_wave": report["wave_index"],
           "created": created, "round_id": control.name, "approval_mode": True,
           "adopted": True, "root": str(root), "control": str(control),
           "config": str(control / "runtime.json"),
           "config_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
           "initial_memory": str(initial_memory),
           "initial_memory_sha256": hashlib.sha256(seed).hexdigest(),
           "source_manifest": str(source_path), "phase1_root": None, "review": None}
    adoption = {key: job[key] for key in (
        "schema", "target", "phase", "objective", "project_target", "model",
        "source_manifest_sha256", "initial_memory_sha256", "legacy_harness_revision",
        "adopting_harness_revision", "historical_code_provenance",
        "historical_source_provenance", "historical_source_manifest_sha256",
        "source_provenance_scope",
        "source_provenance_activation_wave", "created")}
    if ((root / "pending.json").exists()
            or (root / "state.json").read_bytes() != retained_state
            or (root / "campaign-identity.json").read_bytes() != retained_identity):
        raise ValueError("legacy campaign changed during adoption validation")
    control.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + control.name + ".", dir=control.parent))
    try:
        _write(staging / "job.json", job)
        (staging / "runtime.json").write_bytes(wrapper_bytes)
        _write(staging / "adoption.json", adoption)
        _write(staging / "worker.json", {"status": "waiting-approval", "finished": created,
                                          "result": report})
        if _active_campaign_processes(root):
            raise ValueError("active legacy campaign process appeared during adoption validation")
        os.replace(staging, control)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "adopted", "control": str(control), "result": report}


def _require_approval_mode(job):
    if job.get("approval_mode") is not True:
        raise ValueError("campaign was not started in approval mode")


def _proposal_id(proposal):
    payload = json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _approval(job, proposal_id):
    path = _approval_path(job)
    if not path.exists():
        raise ValueError("no curriculum proposal is retained")
    record = _read(path)
    if record.get("proposal_id") != proposal_id:
        raise ValueError("proposal ID does not match the retained proposal")
    return record


def _serialized_approval(function):
    """Serialize public control mutations, including direct Python callers."""
    @wraps(function)
    def locked(job, *args, **kwargs):
        control = Path(job["control"])
        with (control / "campaign.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("campaign control is busy; retry after the active operation") from error
            return function(job, *args, **kwargs)
    return locked


@_serialized_approval
def sample_curriculum(job, *, builder=build_controller):
    """Retain one model-selected proposal without making it executable."""
    _require_approval_mode(job)
    path = _approval_path(job)
    if path.exists() and _read(path).get("status") in {"proposed", "approved", "running"}:
        raise ValueError("resolve the retained curriculum proposal before sampling again")
    proposal = builder(job).propose_wave()
    identity = _proposal_id(proposal)
    projects = [{"project_id": row["spec"]["project_id"],
                 "hypothesis": row["spec"].get("hypothesis", "")}
                for row in proposal.get("wave", {}).get("projects", [])]
    record = {"schema": "ascend-curriculum-approval-v1", "proposal_id": identity,
              "status": "proposed", "phase": job["phase"], "projects": projects,
              "proposal": proposal, "created": time.time()}
    _write(path, record)
    return record


@_serialized_approval
def approve_curriculum(job, project_ids, proposal_id, *, builder=build_controller):
    _require_approval_mode(job)
    record = _approval(job, proposal_id)
    if record["status"] == "approved":
        if tuple(record["selected_project_ids"]) == tuple(project_ids):
            return record
        raise ValueError("proposal is already approved with a different selection")
    if record["status"] != "proposed":
        raise ValueError("only a proposed curriculum can be approved")
    selected = tuple(project_ids)
    terminal = "terminal" in record["proposal"]
    if terminal and selected:
        raise ValueError("terminal approval requires zero project IDs")
    if not terminal and not selected:
        raise ValueError("wave approval requires at least one project ID")
    if not terminal and job["phase"] == 2 and len(selected) != 1:
        raise ValueError("Phase 2 approval requires exactly one project")
    approved = builder(job).approve_wave(record["proposal"], selected)
    record = {**record, "status": "approved", "selected_project_ids": list(selected),
              **({"approved_wave": approved["wave"]} if "wave" in approved else {
                  "approved_terminal": approved["terminal"]}),
              "approved": time.time()}
    _write(_approval_path(job), record)
    return record


@_serialized_approval
def reject_curriculum(job, proposal_id, *, builder=build_controller):
    _require_approval_mode(job)
    record = _approval(job, proposal_id)
    if record["status"] == "rejected":
        return record
    if record["status"] != "proposed":
        raise ValueError("only a proposed curriculum can be rejected")
    builder(job).discard_proposal(record["proposal"])
    record = {**record, "status": "rejected", "rejected": time.time()}
    _write(_approval_path(job), record)
    return record


def _worker_status(result):
    return ("complete" if result.get("status") in {
        "TARGET_REACHED", "PROJECT_LIMIT_REACHED", "SATURATED", "STALLED",
    } else "waiting-approval")


def _annex_options(job):
    if not job.get("adopted"):
        return {}
    return {"external_initial_memory": job["initial_memory"],
            "adoption": str(Path(job["control"]) / "adoption.json")}


@_serialized_approval
def run_approved_curriculum(job, proposal_id, *, builder=build_controller):
    _require_approval_mode(job)
    record = _approval(job, proposal_id)
    if record["status"] == "complete":
        return record
    if record["status"] not in {"approved", "running"}:
        raise ValueError("curriculum proposal is not approved")
    coordinator = builder(job)
    base_wave = record["proposal"]["base_wave_index"]
    if record["status"] == "running":
        current = coordinator.report()
        terminal = record["proposal"].get("terminal")
        reconciled = (current["wave_index"] == base_wave + 1 if terminal is None
                      else current["wave_index"] == base_wave
                      and current["status"] == terminal["status"])
        if reconciled:
            reconcile_feedback = getattr(coordinator, 'reconcile_feedback', None)
            if reconcile_feedback is not None:
                reconcile_feedback()
            record = {**record, "status": "complete", "result": current,
                      "finished": time.time(), "recovered": True}
            _write(_approval_path(job), record)
            _write(Path(job["control"]) / "worker.json", {
                "status": _worker_status(current), "finished": time.time(), "result": current})
            return record
        if current["wave_index"] != base_wave:
            raise ValueError("approved wave completion cannot be reconciled")
    record = {**record, "status": "running", "started": time.time()}
    _write(_approval_path(job), record)
    control = Path(job["control"])
    _write(control / "worker.json", {"status": "running", "started": time.time()})
    try:
        run_proposal = getattr(coordinator, "run_approved_proposal", None)
        result = (run_proposal(record["proposal"]) if run_proposal is not None
                  else coordinator.run_approved_wave())
    except Exception as error:
        _write(control / "worker.json", {"status": "failed", "finished": time.time(),
                                         "error": str(error)})
        raise
    record = {**record, "status": "complete", "result": result, "finished": time.time()}
    _write(_approval_path(job), record)
    _write(control / "worker.json", {"status": _worker_status(result),
                                      "finished": time.time(), "result": result})
    return record


def campaign_worker(job, *, builder=build_controller):
    """One retained worker; infrastructure recovery is an explicit resume."""
    control = Path(job["control"])
    with (control / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write(control / "worker.json", {"status": "running", "started": time.time()})
        try:
            coordinator = builder(job)
            if job.get("continue_target") is not None:
                state_path = Path(job["root"]) / "state.json"
                already_extended = state_path.exists() and _read(state_path).get("project_target") == job["continue_target"]
                if not already_extended:
                    coordinator.continue_campaign(job["continue_target"],
                                                  reopen_terminal=job.get("reopen_terminal", False))
                coordinator.project_target = job["continue_target"]
                job["project_target"] = job.pop("continue_target")
                _write(control / "job.json", job)
                public_path = Path(job["root"]) / "campaign-run.json"
                if public_path.exists():
                    public = _read(public_path)
                    public["project_target"] = job["project_target"]
                    _write(public_path, public)
                elif job.get("adopted"):
                    adoption_path = Path(job["control"]) / "adoption.json"
                    public = _read(adoption_path)
                    public["project_target"] = job["project_target"]
                    _write(adoption_path, public)
                else:
                    raise ValueError("campaign run provenance is missing")
            elif job.get("adopted"):
                adoption_path = control / "adoption.json"
                adoption = _read(adoption_path)
                if adoption.get("project_target") != job["project_target"]:
                    adoption["project_target"] = job["project_target"]
                    _write(adoption_path, adoption)
            result = coordinator.report() if job.get("approval_mode") else coordinator.resume()
            if job.get('retry_feedback_id') and not job.get('phase1_original_retry'):
                from .failure_curriculum import link_retry
                link_retry(job)
            worker_status = _worker_status(result) if job.get("approval_mode") else "complete"
            _write(control / "worker.json", {"status": worker_status, "finished": time.time(),
                                             "result": result})
            return result
        except Exception as error:
            _write(control / "worker.json", {"status": "failed", "finished": time.time(),
                                             "error": str(error)})
            raise


def collector_worker(job, *, once=False, factory=ArtifactCollector, sleeper=time.sleep):
    """No dependency from compute to downloader, including unavailable transport."""
    control = Path(job["control"])
    transport = {}
    if job["target"] == "a3":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path)
        parser.add_argument("--worker-config", type=Path)
        native, _ = parser.parse_known_args(_read(job["config"])["phase1_arguments"])
        from benchmarks.ascend.execution import apply_worker_config, execution_mode
        settings = apply_worker_config(_read(native.config) if native.config else {},
                                      job.get("worker_config") or native.worker_config)
        if execution_mode(settings) == "local":
            transport["execution"] = "local"
        elif native.config is not None:
            transport["cpl_remote"] = settings["cpl_remote"]
    else:
        from benchmarks.ascend.execution import apply_worker_config, execution_mode
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--worker-config', type=Path)
        arguments = _read(job['config']).get('phase1_arguments', []) if job.get('config') else []
        native, _ = parser.parse_known_args(arguments)
        path = job.get('worker_config') or native.worker_config
        if path and execution_mode(apply_worker_config({}, path)) == "local":
            transport["execution"] = "local"
    worker = factory(job["root"], control / "reports", control / "collector.json", **transport)
    while True:
        try:
            result = worker.collect_once()
            _write(control / "collector-status.json", result)
            if (control / "worker.json").exists() and _read(control / "worker.json")["status"] == "complete":
                current = _read(control / "job.json")
                inventory = inventory_campaign(current["root"], artifact_store=control / "reports",
                                               **_annex_options(current))
                destination = control / f"campaign-{current['project_target']}.tar.gz"
                if destination.exists():
                    try:
                        with tarfile.open(destination) as archive:
                            retained = json.load(archive.extractfile("manifest.json"))
                        matches = retained["complete"] and retained["inventory_sha256"] == inventory["inventory_sha256"]
                    except (OSError, tarfile.TarError, KeyError, ValueError):
                        matches = False
                    if not matches:
                        destination = control / f"campaign-{current['project_target']}-{inventory['inventory_sha256'][:12]}.tar.gz"
                        if destination.exists():
                            raise ValueError("checkpoint destination already exists; inspect it before retrying packaging")
                if not destination.exists():
                    package_campaign(current["root"], destination, artifact_store=control / "reports",
                                     **_annex_options(current))
                _write(control / "package-status.json", {"status": "complete", "archive": str(destination)})
                return
        except Exception as error:
            _write(control / "package-status.json", {"status": "pending", "detail": str(error)})
        if once:
            return
        sleeper(300)


def start_screen(job, role, *, runner=subprocess.run):
    control = Path(job["control"])
    name = "rsi-" + hashlib.sha256(str(control).encode()).hexdigest()[:12] + "-" + role
    command = ["screen", "-dmS", name, "-L", "-Logfile", str(control / f"{role}.log"),
               sys.executable, str(Path(__file__).resolve().parents[2] / "run_ascend_campaign.py"),
               "_" + role, "--control", str(control)]
    # Screen creation is idempotent for live sessions, including interrupted CLI startup.
    sessions = runner(["screen", "-ls"], capture_output=True, text=True, check=False).stdout
    if any(line.split()[0].endswith("." + name) for line in sessions.splitlines() if line.strip()):
        return name
    runner(command, check=True)
    return name


def status(job):
    root, control = Path(job["root"]), Path(job["control"])
    state = _read(root / "state.json") if (root / "state.json").exists() else {}
    outcomes = [_read(p) for p in root.glob("waves/*/branches/*/outcome.json")]
    published = len(state.get("prior_outcomes", []))
    attempted = len(list(root.glob("waves/*/branches/*")))
    if state.get("phase") == "optimizer-phase2":
        outcomes = state.get("attempts", [])
        attempted = len(outcomes)
        published = len(state.get("memory_entries", []))
    elapsed = max(time.time() - job["created"], 1) / 3600
    inventory = inventory_campaign(root, artifact_store=control / "reports", **_annex_options(job))
    return describe_statuses({"phase": job["phase"], "target": job["target"], "project_target": job["project_target"],
            "attempted": attempted,
            "verified": sum(o.get("verified") is True for o in outcomes),
            "published": published, "published_per_hour": published / elapsed,
            "artifact_pending": sum(r["status"] != "available" for r in inventory["remote_reports"]),
            "worker": _read(control / "worker.json") if (control / "worker.json").exists() else None})


def main(argv=None, *, screen=start_screen):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    feedback = commands.add_parser('feedback', aliases=['feedback-phase2'], help='import retained failure observations without running agents')
    feedback.add_argument('--feedback-root', type=Path, required=True)
    feedback.add_argument('--source-type', choices=('all', 'phase1', 'phase2', 'optimizer'), default='phase2')
    sources = feedback.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-root', type=Path, action='append')
    sources.add_argument('--dashboard-config', type=Path)
    for command in ("start", "adopt", "resume", "continue", "status", "collect", "package",
                    "sample", "approve", "reject", "run-approved", "_campaign", "_collector"):
        sub = commands.add_parser(command)
        sub.add_argument("--control", type=Path, required=True,
                         help="ignored local settings/queue directory outside the campaign")
        if command == "start":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--target", choices=("a3", "a5"), required=True)
            sub.add_argument("--phase", type=int, choices=(1, 2), required=True)
            sub.add_argument("--objective", required=True)
            sub.add_argument("--initial-memory", type=Path, required=True)
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--worker-config", type=Path)
            sub.add_argument("--project-target", type=int, required=True)
            sub.add_argument("--target-goal", type=Path,
                             help="host-qualified TargetContract JSON; omission lets curriculum author and qualify a new goal")
            sub.add_argument("--phase2-max-seconds", type=float, default=7200)
            sub.add_argument("--practice-max-seconds", type=float, default=1800)
            sub.add_argument("--phase1-root", type=Path)
            sub.add_argument("--review", type=Path)
            sub.add_argument("--approval-mode", action="store_true")
            sub.add_argument('--feedback-root', type=Path)
            sub.add_argument('--failure-id', action='append', default=[])
            sub.add_argument('--retry-feedback-id')
        if command == "adopt":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--objective", required=True)
            sub.add_argument("--initial-memory", type=Path, required=True)
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--model", required=True)
            sub.add_argument("--source-manifest", type=Path, required=True)
            sub.add_argument("--knowledge", choices=("with-kdb", "without-kdb"), required=True)
            sub.add_argument("--profiling", choices=("new", "without-profiling-guidance"), required=True)
            sub.add_argument("--env-file", type=Path)
            sub.add_argument("--legacy-revision", required=True)
        if command == "continue":
            sub.add_argument("--project-target", type=int, required=True)
            sub.add_argument("--reopen-terminal", action="store_true")
        if command in {"status", "collect"}:
            sub.add_argument("--watch", action="store_true")
        if command == "package":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--allow-pending", action="store_true")
        if command in {"approve", "reject", "run-approved"}:
            sub.add_argument("--proposal-id", required=True)
        if command == "approve":
            sub.add_argument("--project-id", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command in {'feedback', 'feedback-phase2'}:
        from .phase2_feedback import FeedbackStore
        store = FeedbackStore(args.feedback_root)
        importers = {'phase1': store.import_phase1, 'phase2': store.import_campaign,
                     'optimizer': store.import_optimizer}
        sources = []
        if args.dashboard_config:
            config = _read(args.dashboard_config)
            sources = [('phase1' if str(row.get('phase')) == '1' else 'phase2', row['root'])
                       for row in config.get('campaigns', []) if str(row.get('phase')) in {'1', '2'}]
            optimizer = config.get('optimizer', {})
            if optimizer.get('root'):
                sources.append(('optimizer', optimizer['root']))
            if args.source_type != 'all':
                sources = [(kind, root) for kind, root in sources if kind == args.source_type]
        else:
            if args.source_type == 'all':
                parser.error('--source-type all requires --dashboard-config')
            sources = [(args.source_type, root) for root in args.source_root]
        print(json.dumps({'reports': [importers[kind](root) for kind, root in dict.fromkeys(sources)]}))
        return
    control = args.control.resolve()
    if args.command == "adopt":
        print(json.dumps(adopt_campaign(args.root, control, args.objective,
                                        args.initial_memory, args.config,
                                        args.legacy_revision, model=args.model,
                                        source_manifest=args.source_manifest,
                                        knowledge=args.knowledge, profiling=args.profiling,
                                        env_file=args.env_file)))
        return
    if args.command == "start":
        from benchmarks.a5kernels.sft_dataset import SourceManifest
        if args.project_target < 1:
            parser.error("--project-target must be positive")
        if args.phase == 2 and (args.phase1_root is None or args.review is None):
            parser.error("phase 2 requires --phase1-root and --review")
        if args.phase == 1 and args.target_goal is not None:
            parser.error("--target-goal is only supported for phase 2")
        if args.phase == 2 and args.approval_mode:
            parser.error("target-driven phase 2 does not support --approval-mode")
        if args.phase == 2 and args.project_target != 1:
            parser.error("new phase 2 starts exactly one goal; use --project-target 1")
        if any(not math.isfinite(value) or value <= 0 for value in (args.phase2_max_seconds, args.practice_max_seconds)):
            parser.error("Phase 2 and practice budgets must be finite positive seconds")
        if (args.failure_id or args.retry_feedback_id) and not args.feedback_root:
            parser.error('feedback selection requires --feedback-root')
        if (args.phase == 1 and args.retry_feedback_id) or (args.phase == 2 and args.failure_id):
            parser.error('failure IDs are Phase 1; retry-feedback-id is Phase 2')
        if args.phase == 2 and args.feedback_root and not args.retry_feedback_id:
            parser.error('Phase 2 feedback requires --retry-feedback-id')
        if args.retry_feedback_id and args.target_goal:
            parser.error('feedback retry already pins its original target')
        root = args.root.resolve()
        if root == control or root in control.parents or control in root.parents:
            parser.error("campaign and control directories must be separate")
        if root.exists() and any(root.iterdir()):
            parser.error("start requires an empty campaign root; use resume")
        config = args.config.resolve()
        settings = _read(config)
        arguments = settings["phase1_arguments"]
        reserved = {"--root", "--objective", "--initial-memory", "--target", "--project-target", "--max-projects", "--max-seconds"}
        if not isinstance(arguments, list) or any(not isinstance(a, str) for a in arguments):
            parser.error("phase1_arguments must be a string list")
        if any(a.split("=", 1)[0] in reserved for a in arguments):
            parser.error("phase1_arguments cannot override campaign identity/budget")
        goal = None
        if args.target_goal is not None:
            from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
            from .phase2_targets import TargetContract
            goal = TargetContract.from_mapping(_read(args.target_goal))
            goal.verify(ContentAddressedArtifacts(Path(settings["target_artifact_store"])),
                        settings["target_runtime"])
            if goal.runtime["target"] != args.target:
                parser.error("goal runtime target differs from --target")
        source_path = Path(settings["source_manifest"])
        source = SourceManifest.load(source_path)
        revision = _harness_revision()
        selection = retry = None
        if args.feedback_root:
            from .failure_curriculum import select_feedback, retry_spec
            try:
                selection = select_feedback(args.feedback_root, args.target, args.failure_id,
                                            retry_id=args.retry_feedback_id)
                if args.retry_feedback_id and selection['selected'][0]['outcome'].get('source_type') == 'phase1':
                    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
                    original = selection['selected'][0]['outcome'].get('original_spec')
                    ProjectSpec.from_mapping(original)
                    retry = dict(source_type='phase1', original_spec=original)
                else:
                    retry = retry_spec(selection, args.target) if args.retry_feedback_id else None
            except (ValueError, KeyError, OSError) as error:
                parser.error(str(error))
        control.mkdir(parents=True, exist_ok=False)
        root.mkdir(parents=True, exist_ok=True)
        (root / "initial-memory.bin").write_bytes(args.initial_memory.read_bytes())
        (root / "source-evidence").mkdir()
        (root / "source-evidence" / "manifest.json").write_bytes(source_path.read_bytes())
        public = {"schema": "ascend-campaign-run-v1", "target": args.target, "phase": args.phase,
                  "objective": args.objective, "project_target": args.project_target,
                  "model": settings["model"], "source_manifest_sha256": source.sha256,
                  "harness_revision": revision, "created": time.time(), "round_id": control.name}
        if goal is not None:
            public.update(target_goal=goal.as_dict(), phase2_max_seconds=args.phase2_max_seconds,
                          practice_max_seconds=args.practice_max_seconds)
        elif args.phase == 2:
            public.update(autonomous_goal=True, phase2_max_seconds=args.phase2_max_seconds,
                          practice_max_seconds=args.practice_max_seconds)
            if args.target == "a3":
                public["optimizer_goal"] = True
        if args.approval_mode:
            public["approval_mode"] = True
        if selection is not None:
            _write(root / 'phase2-feedback.json', selection)
            public.update(feedback_root=str(args.feedback_root.resolve()),
                failure_ids=[r['feedback_id'] for r in selection['selected']],
                feedback_selection_sha256=hashlib.sha256((root / 'phase2-feedback.json').read_bytes()).hexdigest())
            if retry:
                public.update(retry_feedback_id=args.retry_feedback_id, retry_spec=retry)
                if retry.get('source_type') == 'phase1':
                    public['phase1_original_retry'] = True
                    public.pop('optimizer_goal', None)
                    public.pop('autonomous_goal', None)
        _write(root / "campaign-run.json", public)
        job = {**public, "root": str(root), "control": str(control), "config": str(config),
               "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
               "phase1_root": str(args.phase1_root.resolve()) if args.phase1_root else None,
               "review": str(args.review.resolve()) if args.review else None}
        if args.worker_config is not None:
            job["worker_config"] = str(args.worker_config.resolve(strict=True))
            job["worker_config_sha256"] = hashlib.sha256(args.worker_config.read_bytes()).hexdigest()
        if (job.get("optimizer_goal") or args.phase == 1 and settings.get("automatic_remediation")) and settings.get("optimizer_worker_config"):
            job["optimizer_worker_config_sha256"] = hashlib.sha256(
                Path(settings["optimizer_worker_config"]).read_bytes()).hexdigest()
        if goal is not None:
            source_store = ContentAddressedArtifacts(Path(settings["target_artifact_store"]))
            pinned_store = ContentAddressedArtifacts(control / "target-artifacts")
            reference = source_store.read_json(goal.reference_sha256)
            digests = [goal.reference_sha256, reference["source_sha256"],
                       *reference["artifacts"].values(),
                       *(row["evidence_sha256"] for row in reference["timings"])]
            for digest in digests:
                pinned_store.put_bytes(source_store.read_bytes(digest))
            job["target_artifact_store"] = str(pinned_store.root)
        _write(control / "job.json", job)
    else:
        job = _read(control / "job.json")
    if args.command in {"start", "resume", "continue"}:
        with (control / "campaign.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("campaign is already running; use status")
        if args.command == "continue":
            if job.get("target_goal") is not None or job.get("autonomous_goal"):
                parser.error("target-driven budgets are fixed; start a new campaign to change them")
            adoption = (_read(control / "adoption.json") if job.get("adopted")
                        and (control / "adoption.json").exists() else {})
            reconciling = (job.get("adopted") is True
                           and args.project_target == job["project_target"]
                           and adoption.get("project_target") != args.project_target)
            if args.project_target < job["project_target"] or (
                    args.project_target == job["project_target"] and not reconciling):
                parser.error("continuation must increase project target")
            job.update(continue_target=args.project_target, reopen_terminal=args.reopen_terminal)
            _write(control / "job.json", job)
        _write(control / "worker.json", {"status": "queued"})
        # Failure to start the optional collector must not suppress compute.
        try:
            screen(job, "collector")
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"collector not started: {error}", file=sys.stderr)
        print(json.dumps({"session": screen(job, "campaign"), "control": str(control)}))
    elif args.command == "_campaign":
        print(json.dumps(campaign_worker(job)))
    elif args.command == "_collector":
        collector_worker(job)
    elif args.command == "collect":
        collector_worker(job, once=not args.watch)
    elif args.command == "package":
        print(json.dumps(package_campaign(job["root"], args.output,
                         artifact_store=control / "reports", allow_pending=args.allow_pending,
                         **_annex_options(job))))
    elif args.command == "sample":
        print(json.dumps(sample_curriculum(job)))
    elif args.command == "approve":
        print(json.dumps(approve_curriculum(job, args.project_id, args.proposal_id)))
    elif args.command == "reject":
        print(json.dumps(reject_curriculum(job, args.proposal_id)))
    elif args.command == "run-approved":
        print(json.dumps(run_approved_curriculum(job, args.proposal_id)))
    else:
        while True:
            print(json.dumps(status(job)), flush=True)
            if not args.watch:
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
