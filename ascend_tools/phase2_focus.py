"""Frozen five-case Phase 2 scoring, retaining full baseline qualification."""
import argparse
import gzip
import io
import json
from pathlib import Path
import statistics
import tarfile
import uuid

from .baselines import describe, digest, file_hash, validate_receipt
from .baselines.projects import prepare_project, _LAUNCHER

FOCUSED_LAUNCHER = _LAUNCHER.replace(
    'from ascend_tools.baselines.project_verifier import main',
    'from ascend_tools.phase2_focus import main')


def select_focus(receipt):
    """Rank qualified device-call medians, with a stable case-ID tie break."""
    expected = {**describe(receipt['workload']), 'target': receipt['target'],
                'runtimeFingerprint': receipt['runtime']['fingerprint']}
    validate_receipt(receipt, expected)
    timing = receipt['timing']
    if (timing.get('method') != 'synchronized-call-wall-latency' or
            timing.get('warmup') != 5 or timing.get('repeats') != 20 or
            any(len(row['seconds']) != 20 for row in timing['samples'])):
        raise ValueError('Focus selection requires the qualified 5-warmup/20-sample timing protocol')
    ranked = sorted(({'id': row['id'], 'medianSeconds': statistics.median(row['seconds'])}
                     for row in timing['samples']), key=lambda row: (-row['medianSeconds'], row['id']))
    if len(ranked) < 5:
        raise ValueError('Five qualified cases required')
    focus = dict(format=1, kind='slowest-five-median', caseIds=[row['id'] for row in ranked[:5]],
                 cases=ranked[:5], fullCaseCount=len(ranked), fullCaseHash=receipt['caseHash'],
                 qualificationHash=digest(receipt))
    return {**focus, 'scopeHash': digest(focus)}


def validate_focus(contract, focus):
    from .baselines import project_verifier as verifier
    verifier.validate_contract(contract)
    if focus != select_focus(contract['qualification']):
        raise ValueError('Frozen five-case selection differs from qualified timing evidence')
    return focus


def focused_files(focus):
    return {'focus.json': json.dumps(focus, sort_keys=True).encode(),
            'ascend_tools/phase2_focus.py': Path(__file__).read_bytes()}


def prepare_focused_project(workload, target, root, receipt, identity, python='python3', project_id=None):
    focus = select_focus(receipt)
    project = prepare_project(workload, target, root, receipt, identity, python, project_id)
    launcher, archive = map(Path, project['verifier']['files'])
    with tarfile.open(archive) as tar:
        files = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    files.update(focused_files(focus))
    with archive.open('wb') as output, gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o444
                tar.addfile(info, io.BytesIO(data))
    launcher.write_text(FOCUSED_LAUNCHER.replace('ARCHIVE_HASH', file_hash(archive)))
    project['benchmark_focus'] = focus
    listing = ', '.join(f"{row['id']} ({row['medianSeconds']:.6g}s)" for row in focus['cases'])
    instruction = (f'Optimize {workload} in {"Triton-Ascend" if target == "a3" else "Catlass DSL"}. '
                   'Export run(*args, **attrs) with the bundled PyTorch argument/output contract. '
                   f'The frozen iteration and final-scoring scope is exactly five cases: {listing}. '
                   'These are the slowest median synchronized device-call cases from this target\'s '
                   'full qualification. Prioritize them; the verifier checks and scores only this set. '
                   'Preserve input values, output structure/dtypes and NPU placement. '
                   'The complete reference pack remains bundled for inspection.')
    project['goal'] = instruction
    project['performance_goal'] = ('Pass all five selected numerical cases and exceed geometric-mean '
                                   'speedup 1 over fresh paired PyTorch measurements on those same cases.')
    (Path(project['source']) / 'FOCUS.md').write_text(instruction + '\n')
    (Path(project['source']) / 'focus.json').write_text(json.dumps(focus, indent=2) + '\n')
    return project


def evaluate_focused(contract, focus, function, *, device='cpu', performance=False, progress=None, **options):
    from .baselines import project_verifier as verifier
    validate_focus(contract, focus)
    result = verifier.evaluate(contract['workload'], function, device=device, performance=performance,
                               case_ids=focus['caseIds'], progress=progress, **options)
    rows = result['cases']
    complete = (sorted(row['id'] for row in rows) == sorted(focus['caseIds']) and
                all(row.get('passed') is True for row in rows))
    result.update(evaluationScope=focus, complete=complete, passed=complete and (
        not performance or result['geometricMeanSpeedup'] > 1))
    return result


def verify_focused(contract, focus, candidate, artifacts, mode):
    from .baselines import project_verifier as verifier
    validate_focus(contract, focus)
    torch = verifier.torch
    if contract['identity'].get('torch') != torch.__version__:
        raise ValueError('Observed NPU runtime differs')
    import torch_npu
    observed = {'torch': torch.__version__, 'torchNpu': torch_npu.__version__,
                'deviceName': torch.npu.get_device_name(0)}
    if any(contract['identity'].get(key) != value for key, value in observed.items()):
        raise ValueError('Observed NPU runtime differs')
    function, source_hash = verifier.load_candidate(candidate)
    directory = Path(artifacts) / uuid.uuid4().hex; directory.mkdir(parents=True, exist_ok=False)
    common = dict(candidateHash=source_hash, workload=contract['workload'], target=contract['target'],
                  sourceHash=contract['expected']['sourceHash'], caseHash=contract['expected']['caseHash'],
                  verifierHash=contract['expected']['verifierHash'], qualificationHash=digest(contract['qualification']),
                  runtimeFingerprint=contract['identity']['fingerprint'], language=contract['language'],
                  languageAttested=False, evaluationScope=focus)
    partial = {}
    def progress(value):
        partial.update(value)
        verifier.atomic(directory / 'result.json', dict(common, passed=False, complete=False, **value))
    try:
        result = evaluate_focused(contract, focus, function, device='npu:0', performance=mode == 'performance',
                                  warmup=contract['warmup'], repeats=contract['repeats'], progress=progress)
        if file_hash(candidate) != source_hash:
            raise ValueError('Candidate source changed during verification')
    except Exception as error:
        result = dict(partial, passed=False, complete=False, targetEvidence=bool(partial.get('cases')),
                      error=f'{type(error).__name__}: {error}')
    result.update(common)
    verifier.atomic(directory / 'result.json', result)
    return {**result, 'evidence': str(directory / 'result.json')}


def main(contract_path):
    from .baselines import project_verifier as verifier
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True); parser.add_argument('--artifacts', required=True)
    parser.add_argument('--mode', choices=['correctness', 'performance'], required=True)
    args = parser.parse_args()
    try:
        result = verify_focused(json.loads(Path(contract_path).read_text()),
                                json.loads(Path(contract_path).with_name('focus.json').read_text()),
                                args.candidate, args.artifacts, args.mode)
    except Exception as error:
        result = {'passed': False, 'targetEvidence': False, 'error': f'{type(error).__name__}: {error}'}
        directory = Path(args.artifacts) / uuid.uuid4().hex; directory.mkdir(parents=True, exist_ok=False)
        verifier.atomic(directory / 'result.json', result); result['evidence'] = str(directory / 'result.json')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['passed'] else 1
