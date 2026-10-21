"""Bounded local Catlass proof, launched inside an activated A5 environment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from benchmarks.ascend.execution import apply_worker_config
from benchmarks.ascend.local_catlass import LocalSessionAdapter, LocalProfileBackend
from benchmarks.a5kernels.runner import A5KernelRunner
from benchmarks.a5kernels.protocol import RunRequest
from benchmarks.a5kernels.profiling import (CampaignKind, ProfileRequest, TimingCommand,
                                         CaptureCommand, ProfileMetric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    config = apply_worker_config({}, args.worker_config)
    if config.get('execution') != 'local':
        raise ValueError('proof requires local execution')
    backend = LocalSessionAdapter(config)
    runner = A5KernelRunner(backend)
    request = RunRequest('catlass-dsl', length=32)
    plan = backend.prepare_execution(runner.prepare(request, attempt_id='local-proof'))
    verified = runner.run_plan(request, plan)
    if not verified.passed:
        raise AssertionError('local Catlass kernel failed host verification')
    measurements = LocalProfileBackend(config, evidence_directory=args.root / 'measurements')
    profile = ProfileRequest.from_execution_plan(plan, implementation='catlass',
        expected_kernel='*', device=config['physical_device'], warm_up=1, launch_count=3)
    timing = measurements.time_sample(TimingCommand(CampaignKind.INTERMEDIATE, profile, 'local-timing'))
    basic = measurements.capture(CaptureCommand(CampaignKind.INTERMEDIATE, profile, ProfileMetric.BASIC_INFO, 'local-basic'))
    if not basic.exported_kernels:
        raise AssertionError('local profiler did not identify the executed kernel')
    pipe = measurements.capture(CaptureCommand(CampaignKind.INTERMEDIATE, profile,
        ProfileMetric.PIPE_UTILIZATION, 'local-pipe', basic.exported_kernels[0]))
    result = {'status': 'passed', 'execution': 'local', 'verified': asdict(verified),
              'runtime_provenance': backend.runtime_provenance,
              'timing': asdict(timing), 'basic': asdict(basic), 'pipe': asdict(pipe)}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'proof.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
