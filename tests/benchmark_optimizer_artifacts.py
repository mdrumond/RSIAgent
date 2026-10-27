"""Run with python -m tests.benchmark_optimizer_artifacts (host-only, no provider)."""
import argparse
import tempfile
import time
from pathlib import Path

from optimizer.jobs import JobStore, atomic, sha


def measure(root, count, memory_mib, *, legacy=False):
    store = JobStore(root)
    job = store.create('def forward(x): return x', 'forward', {},
                       {'memory': 'x' * (memory_mib * 1024 * 1024)})['job_id']
    started = time.monotonic()
    if legacy:
        # Exact pre-batch path: two submission authentications and one full
        # event-chain rewrite for every artifact.
        for index in range(count):
            data = f'value,{index}\n'.encode()
            store.get(job)
            digest = sha(data)
            path = store.artifact_path(job, digest)
            path.parent.mkdir(exist_ok=True)
            atomic(path, data)
            store.add_event(job, 'artifact', dict(name=f'{index}.csv', sha256=digest, size=len(data)))
    else:
        store.put_artifacts(job, ((f'{i}.csv', f'value,{i}\n'.encode()) for i in range(count)))
    elapsed = time.monotonic() - started
    assert len(store.artifacts(job)) == count
    print(f'{"legacy" if legacy else "batch"}: {count} artifacts, {memory_mib} MiB submission, '
          f'{elapsed:.3f}s, {count / elapsed:.1f} artifacts/s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=10000)
    parser.add_argument('--legacy-count', type=int, default=100)
    parser.add_argument('--memory-mib', type=int, default=56)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='optimizer-artifact-benchmark-') as temporary:
        root = Path(temporary)
        measure(root / 'legacy', args.legacy_count, args.memory_mib, legacy=True)
        measure(root / 'batch-small', args.legacy_count, args.memory_mib)
        measure(root / 'batch-full', args.count, args.memory_mib)


if __name__ == '__main__':
    main()
