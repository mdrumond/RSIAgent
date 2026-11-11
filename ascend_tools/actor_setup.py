"""Reusable actor scratch and compiler-cache defaults inside the selected runtime."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def runtime_identity(binding):
    def stamp(path):
        path = Path(path).resolve()
        try:
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns]
        except OSError:
            return [str(path), None]

    packages = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "catlass", "catlass-dsl"):
        try:
            dist = importlib.metadata.distribution(name)
            record = (dist.read_text("RECORD") or "") + (dist.read_text("direct_url.json") or "")
            packages[name] = [dist.version, str(dist.locate_file("")),
                              hashlib.sha256(record.encode()).hexdigest(),
                              [stamp(dist.locate_file(file)) for file in (
                                  "triton/compiler/compiler.py", "triton/backends/ascend/compiler.py",
                                  "triton/_C/libtriton.so", "triton_ascend/__init__.py", "catlass/__init__.py")
                               if Path(dist.locate_file(file)).exists()]]
        except importlib.metadata.PackageNotFoundError:
            continue
    tools = {name: stamp(path) for name in ("bisheng", "bishengir-compile", "hivmc", "ptxas")
             if (path := shutil.which(name))}
    return {"binding": binding, "python": [sys.version, stamp(sys.executable)],
            "machine": platform.machine(), "packages": packages, "compilers": tools,
            "cann": {key: os.environ.get(key) for key in
                     ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH")}}


def prepare(config, env=None):
    env = dict(os.environ if env is None else env)
    identity = runtime_identity(config["binding"])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(config["cache_root"]) / "v1" / key
    scratch = Path(config["scratch"])
    defaults = {"TRITON_CACHE_DIR": str(root / "triton"),
                "RSI_SCRATCH_DIR": str(scratch), "RSI_SETUP_REPORT": str(scratch / "setup.json"),
                "RSI_SOURCE_DIR": os.getcwd()}
    for name, value in defaults.items():
        env.setdefault(name, value)
    # Catlass's runtime-owned cache is already initialized with bc_compile.
    # Leave that setting intact rather than creating an uninitialized cache.
    for name in ("TRITON_CACHE_DIR", "RSI_SCRATCH_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    report = Path(env["RSI_SETUP_REPORT"])
    report.parent.mkdir(parents=True, exist_ok=True)
    data = {"identity": identity, "environment": {name: env[name] for name in defaults},
            "catlass_cache": env.get("CATLASS_DSL_CACHE_DIR"), "logical_device": 0,
            "python": sys.executable,
            "starter_argv": [sys.executable, "-c", "import torch, torch_npu; "
                             "x=torch.arange(4, device='npu:0'); print((x+1).cpu())"]}
    try:
        retained = json.loads(report.read_text())
    except (OSError, ValueError):
        retained = None
    if retained != data:
        fd, temporary = tempfile.mkstemp(prefix=".setup-", dir=report.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, sort_keys=True)
            os.replace(temporary, report)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return env


def main():
    config = json.loads(sys.argv[1])
    env = prepare(config)
    os.execvpe("bash", ["bash", "-c", sys.argv[2]], env)


if __name__ == "__main__":
    main()
