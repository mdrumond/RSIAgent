"""Time-bounded tiled matmul-add smoke using the production optimizer."""
import argparse
import json
from pathlib import Path

from coding.pi import runtime_preflight
from coding.pi.budget import allocation_lock
from optimizer.agent import OptimizerAgent
from optimizer.jobs import JobStore, canonical, sha
from optimizer.worker import load_config, run_worker
from .report import usage_totals
from .tasks import MATMUL_SHAPES

REFERENCE = """import torch
def forward(a, b, c):
    return torch.matmul(a.float(), b.float()) + c
"""
TASK = """Implement D=A@B+C using tiled Ascend C and the production optimizer_kernel ABI.
A and B are contiguous float16; C, accumulation and output are float32.
Fixed (M,N,K) cases: (64,64,64), (129,193,257), (257,321,193).
Use 64x64 output tiles and reduction chunks of 32; handle all partial tiles and
reduction tails. The larger cases require 12 and 30 output tiles, and 9 and 7
reduction chunks. Use bounded tile buffers and an explicit kernel task type.
The host's INPUTi/OUTPUTi constants bind each frozen case. Choose launch geometry
that handles every case; guard unused blocks or use a grid-stride tile loop.
Only host numerical verification establishes success. Compile feedback may
guide repairs. Never change the frozen reference, representatives, or tolerance.
"""


def signatures():
    return [{"args": [{"shape": [m, k], "dtype": "float16"},
                      {"shape": [k, n], "dtype": "float16"},
                      {"shape": [m, n], "dtype": "float32"}]}
            for m, n, k in MATMUL_SHAPES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path,
                        help="Legacy accounting path; no spend limit is enforced")
    parser.add_argument("--root", type=Path, required=True, help="Ignored local run directory")
    parser.add_argument("--window", required=True, help="Stable incremental allocation name; reuse on resume")
    parser.add_argument("--budget-usd", type=float, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--knowledge-file", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    runtime_preflight()
    config = load_config(args.worker_config)
    for key in ("cpl_remote", "validation_wrapper", "guide"):
        if not Path(config[key]).is_file():
            parser.error(f"configured {key} is unavailable")
    report = run_smoke(config=config, budget_ledger=args.budget_ledger,
                       root=args.root, window=args.window, budget_usd=args.budget_usd,
                       knowledge_file=args.knowledge_file, check_only=args.check_only, max_seconds=args.max_seconds)
    print(json.dumps(report))
    return 0 if args.check_only or report["host_verified"] else 2


def run_smoke(*, config, budget_ledger=None, root, window, budget_usd=2,
              knowledge_file=None, check_only=False, transport_factory=None, max_seconds=1200):
    """Run the fixed smoke with caller-validated runtime/transport configuration.

    Custom transports are trusted host code, injected only through this Python
    API. The CLI retains the checked production route and its config validator.
    """
    if type(max_seconds) is not int or max_seconds <= 0:
        raise ValueError("max_seconds must be a positive integer")
    runtime_preflight()
    root = Path(root)
    knowledge_file = Path(knowledge_file) if knowledge_file else None
    if transport_factory is not None:
        if not config.get("lock_directory") or not config.get("transport_identity"):
            raise ValueError("injected transport requires lock_directory and transport_identity")
    # The provider/model and fixed case count are part of this smoke's contract.
    if config.get("model_profile", "deepseek-flash") != "deepseek-flash":
        raise ValueError("this smoke requires the registered deepseek-flash profile")
    config = {**config, "max_cases": 3, "model_profile": "deepseek-flash"}
    if check_only:
        return {"status": "local-preflight-passed", "remote_validated": False,
                "max_seconds": max_seconds}
    guide = Path(config["guide"]).read_text() + "\n" + TASK
    if knowledge_file:
        guide += "\nStarting knowledge:\n" + knowledge_file.read_text()
    body = {"files": {}, "memory_entries": []}
    memory = {**body, "snapshot_id": sha(canonical(body))}
    with allocation_lock(root):
        store = JobStore(root)
        row = store.create(REFERENCE, "forward", signatures(), memory,
                           name="Pi tiled matmul-add smoke", idempotency_key=window,
                           max_seconds=max_seconds)
        agent = OptimizerAgent(guide=guide)
        failure = None
        try:
            result = run_worker(root, row["job_id"], config, agent=agent,
                                transport_factory=transport_factory)
        except Exception as exc:
            failure = type(exc).__name__
            result = store.get(row["job_id"])
        events = store.events(row["job_id"])
        records_by_id = {}
        for event in events:
            if event["kind"] not in {"model-pi-provider", "model-pi-provider-request"}:
                continue
            data = json.loads(store.read_artifact(row["job_id"], event["payload"]["sha256"]))
            rid = data["reservation_id"]
            if event["kind"] == "model-pi-provider-request":
                records_by_id[rid] = {"usage": None, "status": "unknown"}
            else:
                records_by_id[rid] = data
        records = list(records_by_id.values())
        state = next((event["payload"] for event in reversed(events)
                      if event["kind"] == "optimizer-state"), {})
        # A verified host result still needs manual inspection of the generated
        # source for tiling. Do not label a numerical pass as source inspection.
        report = {"status": result["status"], "host_verified": bool(state.get("best")),
                  "source_inspection": "pending", "usage": usage_totals(records),
                  "evaluated_candidates": result["attempts"], "failure_type": failure}
        (root / "smoke-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


if __name__ == "__main__":
    raise SystemExit(main())
