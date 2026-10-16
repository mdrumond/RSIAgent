"""Explicit phase campaigns, independent artifact collection, portable checkpoints."""
import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
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
        built = (build_native_adaptive_phase1 if job["target"] == "a3"
                 else build_adaptive_phase1)(**options)
        if job["phase"] == 2 and job["target"] == "a3":
            identity_path = Path(job["root"]) / "campaign-identity.json"
            if identity_path.exists():
                current = _read(identity_path)
                for key, value in snapshot.destination.source_identity.items():
                    if key != "schema" and current.get(key) != value:
                        raise ValueError("native treatment changed reviewed Phase 1: " + key)
        captured.append(getattr(built, "coordinator", built))
        # The old CLI calls run() after composing. Capture without executing it.
        return type("Deferred", (), {"run": lambda self: {"deferred": True}})()

    arguments = ["run", "--root", job["root"], "--objective", job["objective"],
                 "--initial-memory", str(Path(job["root"]) / "initial-memory.bin"),
                 "--project-target", str(job["project_target"]), *settings["phase1_arguments"]]
    with redirect_stdout(io.StringIO()):
        target_main(["--target", job["target"], *arguments], native_builder=capture,
                    a5_main=lambda argv: a5_main(argv, builder=capture))
    return captured[0]


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
                public = _read(public_path)
                public["project_target"] = job["project_target"]
                _write(public_path, public)
            result = coordinator.resume()
            _write(control / "worker.json", {"status": "complete", "finished": time.time(),
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
        native, _ = parser.parse_known_args(_read(job["config"])["phase1_arguments"])
        if native.config is not None:
            transport["cpl_remote"] = _read(native.config)["cpl_remote"]
    worker = factory(job["root"], control / "reports", control / "collector.json", **transport)
    while True:
        try:
            result = worker.collect_once()
            _write(control / "collector-status.json", result)
            if (control / "worker.json").exists() and _read(control / "worker.json")["status"] == "complete":
                current = _read(control / "job.json")
                inventory = inventory_campaign(current["root"], artifact_store=control / "reports")
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
                    package_campaign(current["root"], destination, artifact_store=control / "reports")
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
    elapsed = max(time.time() - job["created"], 1) / 3600
    inventory = inventory_campaign(root, artifact_store=control / "reports")
    return {"phase": job["phase"], "target": job["target"], "project_target": job["project_target"],
            "attempted": len(list(root.glob("waves/*/branches/*"))),
            "verified": sum(o.get("verified") is True for o in outcomes),
            "published": published, "published_per_hour": published / elapsed,
            "artifact_pending": sum(r["status"] != "available" for r in inventory["remote_reports"]),
            "worker": _read(control / "worker.json") if (control / "worker.json").exists() else None}


def main(argv=None, *, screen=start_screen):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "resume", "continue", "status", "collect", "package", "_campaign", "_collector"):
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
            sub.add_argument("--project-target", type=int, required=True)
            sub.add_argument("--phase1-root", type=Path)
            sub.add_argument("--review", type=Path)
        if command == "continue":
            sub.add_argument("--project-target", type=int, required=True)
            sub.add_argument("--reopen-terminal", action="store_true")
        if command in {"status", "collect"}:
            sub.add_argument("--watch", action="store_true")
        if command == "package":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args(argv)
    control = args.control.resolve()
    if args.command == "start":
        from benchmarks.a5kernels.sft_dataset import SourceManifest
        if args.project_target < 1:
            parser.error("--project-target must be positive")
        if args.phase == 2 and (args.phase1_root is None or args.review is None):
            parser.error("phase 2 requires --phase1-root and --review")
        root = args.root.resolve()
        if root == control or root in control.parents or control in root.parents:
            parser.error("campaign and control directories must be separate")
        if root.exists() and any(root.iterdir()):
            parser.error("start requires an empty campaign root; use resume")
        config = args.config.resolve()
        settings = _read(config)
        arguments = settings["phase1_arguments"]
        reserved = {"--root", "--objective", "--initial-memory", "--target", "--project-target", "--max-projects"}
        if not isinstance(arguments, list) or any(not isinstance(a, str) for a in arguments):
            parser.error("phase1_arguments must be a string list")
        if any(a.split("=", 1)[0] in reserved for a in arguments):
            parser.error("phase1_arguments cannot override campaign identity/budget")
        source_path = Path(settings["source_manifest"])
        source = SourceManifest.load(source_path)
        control.mkdir(parents=True, exist_ok=False)
        root.mkdir(parents=True, exist_ok=True)
        (root / "initial-memory.bin").write_bytes(args.initial_memory.read_bytes())
        (root / "source-evidence").mkdir()
        (root / "source-evidence" / "manifest.json").write_bytes(source_path.read_bytes())
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True).strip()
        public = {"schema": "ascend-campaign-run-v1", "target": args.target, "phase": args.phase,
                  "objective": args.objective, "project_target": args.project_target,
                  "model": settings["model"], "source_manifest_sha256": source.sha256,
                  "harness_revision": revision, "created": time.time(), "round_id": control.name}
        _write(root / "campaign-run.json", public)
        job = {**public, "root": str(root), "control": str(control), "config": str(config),
               "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
               "phase1_root": str(args.phase1_root.resolve()) if args.phase1_root else None,
               "review": str(args.review.resolve()) if args.review else None}
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
            if args.project_target <= job["project_target"]:
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
                         artifact_store=control / "reports", allow_pending=args.allow_pending)))
    else:
        while True:
            print(json.dumps(status(job)), flush=True)
            if not args.watch:
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
