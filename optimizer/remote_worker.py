"""Remote executable protocol; large frozen tensor bundles stay on the worker."""
import argparse
import copy
import json
from pathlib import Path
import tarfile

from .jobs import atomic, canonical, sha


def execute(request):
    import torch
    import torch_npu
    from .measurement import benchmark
    from .native import compile_candidate, describe_case, evaluate, load_reference
    from .reference import freeze_reference, prepare_reference
    from .profiles import profile
    torch.npu.set_device('npu:0')
    directory = Path(request['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    baseline = Path(request['baseline'])
    cache_directory = baseline.parent / 'compiled-cache'
    operation = request['operation']
    if operation == 'prepare':
        prepared = prepare_reference(request['spec'], device='npu:0', max_cases=request.get('max_cases', 24))
        manifest = freeze_reference(prepared, baseline)
        result = {'suite_sha256': manifest['suite_sha256'], 'cases': [], 'manifest': manifest,
                  'remote_reference': str(baseline)}
        for row in prepared.cases:
            args, kwargs = copy.deepcopy((row['args'], row['kwargs']))
            fn = lambda args=args, kwargs=kwargs: prepared.invoke(*args, **kwargs)
            timing = benchmark(fn, directory / (row['case']['id'] + '-torch-timing'))
            result['cases'].append({'case': row['case']['id'], 'contract': describe_case(prepared, row), 'timing': {'torch': timing}})
    elif operation == 'compile':
        result = compile_candidate(baseline, request['source'], directory / 'compilation',
                                   arch=request['arch'], block_count=request['block_count'], cache_directory=cache_directory)
    elif operation == 'evaluate' or request.get('source') is not None:
        result = evaluate(baseline, request['source'], directory / 'evaluation', arch=request['arch'],
                          block_count=request['block_count'], benchmark=operation == 'evaluate', profile=operation == 'profile',
                          cache_directory=cache_directory,
                          profile_policy=request.get('profiling'), diagnostic_only=operation == 'profile')
    else:
        prepared = load_reference(baseline, 'npu:0')
        from .profile_policy import select
        selected, coverage = select(prepared, request.get('profiling'))
        coverage['implementations'] = ['torch']
        result = {'suite_sha256': prepared.manifest['suite_sha256'], 'cases': [], 'profiling_coverage': coverage}
        for row in selected:
            args, kwargs = copy.deepcopy((row['args'], row['kwargs']))
            fn = lambda args=args, kwargs=kwargs: prepared.invoke(*args, **kwargs)
            identity = {'suite_sha256': prepared.manifest['suite_sha256'], 'case': row['case']['id'],
                        'implementation': 'torch'}
            result['cases'].append({'case': row['case']['id'], 'profiling': {'torch': profile(fn,
                directory / (row['case']['id'] + '-torch-profile'), groups=coverage['policy']['groups'], identity=identity)}})
    return result


def publish(directory, result):
    """Explicit allowlist: never transfer tensor payloads, binaries or timelines."""
    atomic(directory / 'result.json', canonical(result))
    files = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.name in {'compact.tar.gz', 'files.json'}:
            continue
        relative = path.relative_to(directory)
        allowed = path.suffix in {'.cpp', '.log'} or path.name in {'result.json', 'command.json', 'build.json', 'manifest.json', 'reference.py', 'timing.json', 'profile.json'}
        allowed |= 'compact' in relative.parts and path.suffix in {'.csv', '.json'}
        # Raw timing rows are compact evidence; profiler timelines stay remote.
        allowed |= path.name.startswith(('op_summary', 'kernel_details', 'task_time')) and path.suffix == '.csv'
        # Diagnostic analysis runs remotely. Digests carry report hashes/paths;
        # normalized tables and raw profiling trees are not portable artifacts.
        diagnostic = any(part.endswith('-profile') for part in relative.parts)
        if diagnostic and ('raw' in relative.parts or 'compact' in relative.parts):
            allowed = False
        if allowed:
            data = path.read_bytes()
            files.append({'path': str(relative), 'sha256': sha(data), 'size': len(data)})
    atomic(directory / 'files.json', canonical(files))
    with tarfile.open(directory / 'compact.tar.gz', 'w:gz') as archive:
        archive.add(directory / 'files.json', arcname='files.json')
        for row in files:
            archive.add(directory / row['path'], arcname=row['path'])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    directory = Path(request['directory'])
    try:
        result = execute(request)
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        publish(directory, {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    publish(directory, result)


if __name__ == '__main__':
    main()
