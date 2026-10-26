"""Known-good native compile and exact host-oracle gate before Phase1 work."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import re
import time
import uuid

from benchmarks.a3kernels.native_smoke import smoke_plan
from benchmarks.a3kernels.phase1_protocol import FailedEvidence
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.ascend.execution import process_deadline


@contextmanager
def _transport_allocation(transport, remaining_seconds):
    # CPL also freezes queue-inclusive wall expiry; bound it to this wave.
    if not hasattr(transport, "remaining_seconds"):
        yield
        return
    previous = transport.remaining_seconds
    transport.remaining_seconds = lambda: min(remaining_seconds(),
        previous() if previous is not None else float("inf"))
    try:
        yield
    finally:
        transport.remaining_seconds = previous


class NativeWaveEnvironmentCheck:
    def __init__(self, config, transports):
        self.config, self.transports = config, tuple(transports)

    def __call__(self, wave_index, remaining_seconds):
        attempt = uuid.uuid4().hex
        root = Path(self.config.root) / "environment-checks" / f"wave-{wave_index:03}" / attempt
        root.mkdir(parents=True)
        report = {"status": "running", "wave_index": wave_index, "workers": []}
        started = time.monotonic()
        def save():
            report["elapsed_seconds"] = time.monotonic() - started
            temporary = root / "report.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(root / "report.json")
        save()
        try:
            for index, transport in enumerate(self.transports):
                seconds = remaining_seconds()
                if seconds <= 0:
                    raise TimeoutError("wave elapsed allocation exhausted before environment check")
                worker = {"worker": index, "runtime": getattr(transport, "runtime_identity", {})}
                report["workers"].append(worker)
                plan = replace(smoke_plan(target=self.config.target, arch=self.config.arch,
                    profile=self.config.execution_profile), request_id="environment-" + attempt,
                    attempt_id=str(index))
                worker.update(execution_id=plan.execution_id, source_fingerprint=plan.source_fingerprint)
                save()
                with process_deadline(time.monotonic() + seconds), _transport_allocation(transport, remaining_seconds):
                    compilation = transport.compile(plan, root / f"worker-{index}")
                    if isinstance(compilation, FailedEvidence):
                        worker["failure"] = {"stage": compilation.stage, "detail": compilation.detail}
                        raise RuntimeError("known-good environment probe did not compile")
                    worker.update(library_sha256=compilation.library_sha256,
                                  compile_stdout=compilation.stdout, compile_stderr=compilation.stderr)
                    if re.search(r"(?:sh|bash):[^\n]*source: (?:not found|command not found)", compilation.stderr):
                        raise RuntimeError("runtime shell setup failed during compilation")
                    if remaining_seconds() <= 0:
                        raise TimeoutError("wave elapsed allocation exhausted during environment compilation")
                    receipt = transport.execute_receipt(compilation)
                    if isinstance(receipt, FailedEvidence):
                        worker["failure"] = {"stage": receipt.stage, "detail": receipt.detail}
                        raise RuntimeError("known-good environment probe did not execute")
                    worker["execution_handle"] = receipt.job_handle
                    if receipt.exit_code != 0:
                        worker.update(execution_stdout=receipt.stdout, execution_stderr=receipt.stderr)
                        raise RuntimeError("environment probe process failed")
                    expected = tuple(sum(xs) for xs in zip(*(t.values for t in plan.inputs)))
                    if tuple(receipt.output) != expected:
                        raise RuntimeError("environment probe failed exact host verification")
                    if remaining_seconds() <= 0:
                        raise TimeoutError("wave elapsed allocation exhausted during environment execution")
                    worker["status"] = "host-verified"
                    save()
            report["status"] = "passed"
            save()
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            save()
            raise ProjectInfrastructureError(
                f"Wave environment check failed; no curriculum/project work admitted. "
                f"Inspect {root / 'report.json'}: {error}") from error
