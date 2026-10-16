"""A2 validation through a configured local remote-agent helper only."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
import uuid

from .tasks import FIXTURES, MATMUL_RUN, task_contract, verify_matmul_outputs


class RemoteUnavailable(RuntimeError):
    pass


class RemoteAgent:
    def __init__(self, config: dict, state_dir: Path, *, timeout=300, deadline=None, runner=subprocess.run):
        self.config, self.state_dir = config, Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.runner = timeout, runner
        self.deadline = deadline or time.monotonic() + 600
        if config.get("remote") != "a2":
            raise ValueError("this experiment requires explicit target a2")
        self.helper = Path(config["access_helper"]).resolve()

    def call(self, *args):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteUnavailable("remote attempt deadline exceeded; inspect any retained job IDs")
        try:
            result = self.runner([sys.executable, str(self.helper), *args],
                                 capture_output=True, text=True, timeout=min(45, remaining))
        except subprocess.TimeoutExpired:
            raise RemoteUnavailable("remote-agent request timed out; inspect existing jobs before resubmitting") from None
        if result.returncode:
            # Do not persist raw output; session/target errors can contain credentials.
            raise RemoteUnavailable("remote-agent request failed; inspect the local helper diagnostics")
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise RemoteUnavailable("remote-agent returned invalid JSON; inspect existing jobs") from None

    def record(self, record):
        with (self.state_dir / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def ready(self):
        health = self.call("health")
        meta = health.get("session_metadata", {})
        if health.get("status") != "ok" or any(meta.get(k) != v for k, v in {
            "execution_mode": "listener", "listener_ssh_transport": "asyncssh",
        }.items()):
            raise RemoteUnavailable("jumper listener/asyncssh registration is required")
        targets = self.call("client", "target", "list")["targets"]
        if not any(t.get("name") == "a2" and t.get("enabled") for t in targets):
            raise RemoteUnavailable("target a2 is unavailable")
        session = self.call("session")
        if "a2" not in session.get("remotes", {}):
            raise RemoteUnavailable("live session has no a2 target")

    def wait(self, job_id, *, transfer=False):
        deadline = min(self.deadline, time.monotonic() + self.timeout)
        while True:
            job = self.call("client", *( ("transfer", "status") if transfer else ("job", "get") ),
                            "--job-id", job_id)
            status = job.get("status")
            if status in ("succeeded", "failed", "cancelled", "canceled"):
                self.record({"id": job_id, "status": status, "result": job.get("result")})
                if status != "succeeded":
                    raise RemoteUnavailable(f"remote job {job_id} ended {status}")
                if not transfer and (job.get("result") or {}).get("exit_code") != 0:
                    raise RemoteUnavailable(f"remote job {job_id} has no successful exit code")
                return job
            if time.monotonic() >= deadline:
                raise RemoteUnavailable(f"remote job {job_id} is still {status}; inspect it before resubmitting")
            time.sleep(1)

    def submit(self, script: str):
        name = f"pi-smoke-{uuid.uuid4().hex}.sh"
        path = self.state_dir / name
        path.write_text(script)
        self.record({"operation": "submit", "script": name, "status": "submitting"})
        job = self.call("client", "script", "submit", "--remote", "a2", "--file", str(path.resolve()),
                        "--remote-path", "/tmp/" + name)
        self.record({"operation": "submit", "script": name, "id": job["id"]})
        return job["id"]

    def upload(self, source, destination):
        destination = PurePosixPath(destination)
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError("upload destination must be workspace-relative")
        job = self.call("client", "transfer", "upload", "--remote", "a2", "--src", str(Path(source).resolve()),
                        "--dst", str(destination))
        self.record({"operation": "upload", "id": job["id"]})
        self.wait(job["id"], transfer=True)

    def log(self, job_id, stream="stdout"):
        return self.call("client", "logs", "get", "--job-id", job_id,
                         "--stream", stream, "--tail", "30").get("content", "")


def inputs(count):
    return ([(i * 17 % 101 - 50) / 8 for i in range(count)],
            [(i * 13 % 67 - 33) / 4 for i in range(count)])


LENGTHS = (1, 7, 32, 257, 4096)


def verify_outputs(rows):
    if not isinstance(rows, list) or len(rows) != len(LENGTHS):
        return False
    for count, row in zip(LENGTHS, rows):
        a, b = inputs(count)
        if not isinstance(row, list) or len(row) != count:
            return False
        if any(type(v) not in (float, int) or not math.isfinite(v) or
               not math.isclose(v, x + y, abs_tol=1e-5, rel_tol=1e-5)
               for v, x, y in zip(row, a, b)):
            return False
    return True


class Validator:
    def __init__(self, run_dir: Path, connection: dict | None, *, task="vector-add", max_validations=3):
        self.root, self.connection = Path(run_dir).resolve(), connection
        self.task, self.contract = task, task_contract(task)
        self.results = []
        self.max_validations = max_validations
        self.deadline = time.monotonic() + 600

    def validate(self):
        if len(self.results) >= self.max_validations:
            raise ValueError("validation cycles exhausted")
        source = self.root / "kernel.cpp"
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 65536:
            raise ValueError("candidate must be a regular file of at most 64 KiB")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result = {"source_sha256": digest, "task": self.task,
                  "status": "infrastructure_unavailable", "passed": False}
        if self.connection is None:
            result["reason"] = "A2 validation disabled for provider-only smoke"
        else:
            try:
                self._remote(source, result)
            except RemoteUnavailable as exc:
                result["reason"] = str(exc)
        self.results.append(result)
        (self.root / "validation.json").write_text(json.dumps(self.results, indent=2))
        return result

    def _remote(self, source, result):
        connection = self.connection
        required = ("cann_env", "python", "arch", "device")
        if any(connection.get(k) is None for k in required):
            raise RemoteUnavailable("A2 toolchain/device preflight configuration is incomplete")
        if type(connection["device"]) is not int or connection["device"] < 0:
            raise ValueError("device must be a nonnegative integer")
        agent = RemoteAgent(connection, self.root / "remote", deadline=self.deadline)
        agent.ready()
        remote_dir = PurePosixPath("pi-investigation") / uuid.uuid4().hex
        # Trusted driver and input fixtures remain outside the candidate's tools.
        driver = self.root / "remote" / "driver.py"
        driver.write_text(driver_source(self.task))
        agent.upload(source, remote_dir / "kernel.cpp")
        agent.upload(driver, remote_dir / "driver.py")
        if self.task == "matmul-add":
            binding = self.root / "remote" / "binding.cpp"
            binding.write_text((FIXTURES / "matmul_add_binding.cpp").read_text())
            agent.upload(binding, remote_dir / "binding.cpp")
        script = validation_script(connection, remote_dir)
        jid = agent.submit(script)
        result["job_id"] = jid
        agent.wait(jid)
        log = agent.log(jid)
        markers = [line.removeprefix("RSI_PI_RESULT=") for line in log.splitlines()
                   if line.startswith("RSI_PI_RESULT=")]
        if len(markers) != 1:
            raise RemoteUnavailable("validation job did not return one result record")
        payload = json.loads(markers[0])
        if payload.get("source_sha256") != result["source_sha256"]:
            raise RemoteUnavailable("remote candidate digest mismatch")
        if payload.get("stage") == "compile_failed":
            result.update(status="compile_failed", diagnostic=payload.get("diagnostic", "")[-4000:])
        elif payload.get("stage") == "runtime_failed":
            result.update(status="runtime_failed", diagnostic=payload.get("diagnostic", "")[-4000:],
                          failed_case=payload.get("active_case"))
        elif payload.get("stage") == "executed":
            rows = payload.get("outputs")
            passed = verify_outputs(rows) if self.task == "vector-add" else verify_matmul_outputs(rows)
            result.update(status="passed" if passed else "incorrect", passed=passed)
            if self.task == "matmul-add":
                from .tasks import matmul_feedback
                result["cases"] = matmul_feedback(rows)
        else:
            raise RemoteUnavailable("remote driver could not initialize the toolchain/runtime")


def validation_script(connection, remote_dir):
    """Use a preflighted running container when the host lacks the toolchain.

    Docker operations run on A2 through the same submitted script. They only
    create a unique temporary candidate directory; no container lifecycle work.
    """
    quote = shlex.quote
    container = connection.get("container")
    workdir = str(remote_dir) if not container else "/tmp/rsi-pi-" + remote_dir.name
    inner = "\n".join((
        "set -e", "source " + quote(connection["cann_env"]),
        "cd " + quote(workdir),
        "export ASCEND_RT_VISIBLE_DEVICES=" + str(connection["device"]),
        quote(connection["python"]) + " driver.py " + quote(connection["arch"]),
    ))
    if not container:
        return "#!/bin/bash\n" + inner + "\n"
    return "\n".join((
        "#!/bin/bash", "set -e",
        "docker exec " + quote(container) + " mkdir -p " + quote(workdir),
        "docker cp " + quote(str(remote_dir) + "/.") + " " + quote(container + ":" + workdir),
        "docker exec " + quote(container) + " /bin/bash -lc " + quote(inner), "",
    ))


DRIVER = '''import hashlib, json, pathlib, shutil, subprocess, sys
root = pathlib.Path.cwd()
digest = hashlib.sha256((root / "kernel.cpp").read_bytes()).hexdigest()
result = {"source_sha256": digest}
try:
    import torch, torch_npu
    compiler = shutil.which("bisheng")
    if not compiler: raise RuntimeError("bisheng unavailable")
    t, n = pathlib.Path(torch.__file__).parent, pathlib.Path(torch_npu.__file__).parent
    includes = [t / "include", t / "include/torch/csrc/api/include", n / "include"]
    libs = [t / "lib", n / "lib"]
    args = [compiler, "-x", "asc", "--npu-arch=" + sys.argv[1], "-shared", "-fPIC", "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
            "-ltorch_npu", "-ltorch", "-lc10", "kernel.cpp", "-o", "candidate.so",
            *["-I" + str(p) for p in includes], *["-L" + str(p) for p in libs]]
    result["stage"] = "compile_failed"
    p = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if p.returncode:
        result.update(stage="compile_failed", diagnostic=(p.stdout + p.stderr)[-4000:])
    else:
        result["stage"] = "runtime_failed"
        torch.ops.load_library(str(root / "candidate.so"))
        torch.npu.set_device(0)
        rows = []
        for count in (1, 7, 32, 257, 4096):
            result["active_case"] = {"length": count}
            a = torch.tensor([(i * 17 % 101 - 50) / 8 for i in range(count)], dtype=torch.float32, device="npu:0")
            b = torch.tensor([(i * 13 % 67 - 33) / 4 for i in range(count)], dtype=torch.float32, device="npu:0")
            output = torch.ops.rsi_pi_smoke.vector_add(a, b)
            torch.npu.synchronize()
            rows.append(output.cpu().tolist())
        result.update(stage="executed", outputs=rows)
except Exception as exc:
    result.setdefault("stage", "infrastructure_unavailable")
    message = str(exc)
    if len(message) > 3500: message = message[:1750] + " ... " + message[-1750:]
    result["diagnostic"] = type(exc).__name__ + ": " + message
print("RSI_PI_RESULT=" + json.dumps(result))
'''


def driver_source(task):
    task_contract(task)
    if task == "vector-add":
        return DRIVER
    start = DRIVER.index("        rows = []")
    end = DRIVER.index('        result.update(stage="executed", outputs=rows)')
    return (DRIVER[:start] + MATMUL_RUN + DRIVER[end:]).replace(
        '"-lc10", "kernel.cpp", "-o"', '"-lc10", "binding.cpp", "-o"')
