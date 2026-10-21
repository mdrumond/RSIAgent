"""Apply shared execution settings to the established A5 launcher arguments."""
from pathlib import Path


def apply_a5_worker_arguments(args, worker, *, source="catlass_source", revision="catlass_revision"):
    if worker is None:
        return
    mode = worker.get("execution", "remote")
    profile = worker.get("execution_profile", worker.get("profile", "bz-a5"))
    if mode == "remote" and profile != "bz-a5":
        raise ValueError("A5 execution requires the bz-a5 remote profile")
    fields = {source: "catlass_source", revision: "catlass_revision", "device": "physical_device",
              "tla_root": "tla_root", "profiling_skill_root": "profiling_skill_root"}
    for argument, key in fields.items():
        if hasattr(args, argument) and key in worker:
            value = worker[key]
            setattr(args, argument, Path(value) if key.endswith("_root") else value)
    if "validation_wrapper" in worker and "tla_root" not in worker:
        args.tla_root = Path(worker["validation_wrapper"]).parent.parent
    if mode == "local":
        args.tla_root = getattr(args, "tla_root", None) or Path.cwd()
        if hasattr(args, "profiling_skill_root"):
            args.profiling_skill_root = args.profiling_skill_root or Path.cwd()
