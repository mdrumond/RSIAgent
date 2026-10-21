"""Bounded native proof for an activated Ascend runtime (no provider requests)."""
import argparse
import json
from pathlib import Path
import subprocess

from benchmarks.ascend.execution import apply_worker_config, preflight_local
from benchmarks.ascend.local_native import LocalNativeCandidateBackend, LocalNativeMeasurements
from benchmarks.a3kernels.native_smoke import smoke_plan
from benchmarks.a3kernels.phase1_protocol import FailedEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = apply_worker_config({}, args.worker_config)
    if config.get("execution") != "local":
        raise ValueError("proof requires explicit local execution")
    original = subprocess.Popen
    class LocalOnlyPopen(original):
        def __init__(self, argv, *positional, **options):
            command = argv if isinstance(argv, (list, tuple)) else (argv,)
            forbidden = {"ssh", "scp", "rsync", "cpl-remote", "catlass-validation.sh", "remote_agent_client.sh"}
            if any(Path(str(arg)).name in forbidden for arg in command):
                raise AssertionError("local proof attempted a remote-access command")
            super().__init__(argv, *positional, **options)
    subprocess.Popen = LocalOnlyPopen
    identity = preflight_local(config)
    target = identity["product"]
    plan = smoke_plan(target=target, arch=config["arch"], profile="local")
    backend = LocalNativeCandidateBackend(config)
    compilation = backend.compile(plan, args.root / "compile")
    if isinstance(compilation, FailedEvidence):
        raise RuntimeError(compilation.detail)
    receipt = backend.execute_receipt(compilation)
    if isinstance(receipt, FailedEvidence):
        raise RuntimeError(receipt.detail)
    expected = tuple(sum(values) for values in zip(*(tensor.values for tensor in plan.inputs)))
    if len(receipt.output) != len(expected) or any(abs(a - b) > 1e-5 for a, b in zip(receipt.output, expected)):
        raise AssertionError("native candidate failed the independent host oracle")
    measurements = LocalNativeMeasurements(compilation=compilation, candidate_backend=backend,
        expected_output=expected, environment_identity=identity, evidence_directory=args.root / "measurements")
    timing = measurements.timing(warm_up=1, launch_count=3)
    profile = measurements.capture("Basic")
    # Exercise the complete user-facing optimizer path with a supplied real
    # kernel instead of a paid provider. Its host protocol freezes and grades.
    from optimizer_worker_proof import ProofAgent
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.worker import run_worker
    from benchmarks.a3kernels.native_smoke import SOURCE
    args.root.mkdir(parents=True, exist_ok=True)
    guide = args.root / 'guide.md'; guide.write_text('Local kernel execution proof')
    store = JobStore(args.root / 'optimizer-jobs')
    body = {'files': {}, 'memory_entries': []}
    snapshot = {**body, 'snapshot_id': sha(canonical(body))}
    signatures = {'signatures': [{'args': [{'shape': [32], 'dtype': 'float32'}] * 3}]}
    job = store.create('def forward(x, y, z): return x+y+z', 'forward', signatures,
                       snapshot, max_seconds=600)
    settings = {**config, 'guide': str(guide), 'max_cases': 1}
    terminal = run_worker(store.root, job['job_id'], settings,
        agent=ProofAgent(SOURCE.replace('void sum_three(', 'void optimizer_kernel(')))
    states = [event['payload'] for event in store.events(job['job_id']) if event['kind'] == 'optimizer-state']
    if terminal['status'] != 'completed' or not states or not any(
            row.get('content', {}).get('status') == 'verified' for row in states[-1].get('history', [])):
        raise AssertionError('local optimizer did not publish a host-verified candidate')
    result = dict(status="passed", execution="local", identity=identity,
                  execution_id=plan.execution_id, library_sha256=compilation.library_sha256,
                  host_verified=True, timing=timing["summary"], profiling=profile["summary"],
                  optimizer_job_id=job['job_id'], optimizer_status=terminal['status'])
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "proof.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
