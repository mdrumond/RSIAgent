"""Portable benchmark inputs, evidence boundaries, and budget allocation safety."""
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.pi_coding.budget import BudgetExceeded, allocation_lock
from benchmarks.pi_coding.config import DEFAULT_KNOWLEDGE, compose_request
from benchmarks.pi_coding.export_evidence import export_bundle
from benchmarks.pi_coding.report import usage_totals


def test_allocation_lease_rejects_overlap_and_releases(tmp_path):
    with allocation_lock(tmp_path):
        with pytest.raises(BudgetExceeded, match="another comparison"):
            with allocation_lock(tmp_path):
                pytest.fail("overlapping lease")
    with allocation_lock(tmp_path):
        pass


@pytest.mark.parametrize('knowledge_mode', ['bundled', 'custom', 'none'])
def test_pair_has_identical_context_and_fresh_starter(tmp_path, monkeypatch, knowledge_mode):
    from benchmarks.pi_coding import compare
    from benchmarks.pi_coding.experiment import starter_source
    monkeypatch.setattr(compare, 'runtime_preflight', lambda: None)
    monkeypatch.setattr(compare, 'LOCAL', tmp_path)
    monkeypatch.setattr(compare.Validator, 'validate', lambda _: {'passed': True, 'status': 'passed'})
    monkeypatch.setattr(compare, 'provider_preflight', lambda *_: {'passed': True})
    seen = []
    def run(args):
        seen.append(args)
        assert 'TODO' in starter_source(args.task)
        assert 'DataCopyPad' not in starter_source(args.task)
        return {'backend': args.backend, 'run': args.backend, 'status': 'compile_failed', 'stop_reason': 'revision_limit', 'requests': [], 'validations': [], 'writes': 0, 'seconds': 0}
    monkeypatch.setattr(compare, 'run_experiment', run)
    connection = tmp_path / 'connection.json'
    connection.write_text('{}')
    extra = []
    knowledge = DEFAULT_KNOWLEDGE.read_text()
    if knowledge_mode == 'custom':
        path = tmp_path / 'knowledge.txt'
        path.write_text('Relevant portable guidance.')
        extra = ['--knowledge-file', str(path)]
        knowledge = path.read_text()
    elif knowledge_mode == 'none':
        extra = ['--no-knowledge']
        knowledge = ''
    assert compare.main(['--connection', str(connection), '--budget-usd', '1', *extra, 'TASK']) == 2
    assert len(seen) == 2
    assert seen[0].prompt == seen[1].prompt == compose_request('TASK', knowledge)
    assert seen[0].provider == seen[1].provider == 'deepseek'
    assert seen[1].pi_workflow == 'persistent'
    assert seen[0].max_revisions == seen[1].max_revisions == 12
    report = json.loads(next(tmp_path.glob('comparisons/*/comparison.json')).read_text())
    assert report['knowledge']['sha256'] == hashlib.sha256(knowledge.encode()).hexdigest()
    assert report['starter_sha256'] == hashlib.sha256(starter_source('matmul-add').encode()).hexdigest()
    assert report['implementation']['implementation_sha256']


def test_new_pair_requires_explicit_budget():
    from benchmarks.pi_coding.compare import main
    with pytest.raises(SystemExit):
        main(['--connection', 'unused.json'])


def test_evidence_export_drops_untrusted_text_and_nested_fields():
    secret = 'sk-' + 'SENSITIVE' * 8
    private = '/home/private-user/connection.json'
    report = {'prompt': secret, 'connection': private, 'provider_preflight': {'requests': [
        {'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'secret': secret}, 'body': secret}]}}
    result = {'backend': 'pi', 'provider': 'deepseek', 'model': secret, 'status': 'passed', 'stop_reason': private,
              'requests': [{'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 5, 'secret': secret}}, 'prompt': secret}],
              'validations': [{'status': 'passed', 'passed': True, 'source_sha256': 'a'*64, 'job_id': private,
                               'diagnostic': secret, 'cumulative_usage': {'total_tokens': 30, 'secret': secret},
                               'cases': [{'shape_mnk': [64,64,64], 'passed': True, 'max_abs_error': 0, 'private': private}]}],
              'generation': {'reasoning': private, 'max_tokens': 16384}, 'secret': secret}
    bundle = export_bundle(report, [result])
    data = json.dumps(bundle)
    assert secret not in data and private not in data
    assert 'diagnostic' not in data and 'job_id' not in data
    assert bundle['arms'][0]['usage']['total_tokens'] == 30
    assert bundle['preflight_usage']['total_tokens'] == 3
    assert bundle['arms'][0]['model'] == 'unknown'


def test_published_evidence_totals_and_source_digest():
    root = Path(__file__).resolve().parents[1] / 'benchmarks/pi_coding/evidence/2026-10-16'
    bundle = json.loads((root / 'evidence.json').read_text())
    rsi, pi = bundle['arms']
    assert [rsi['backend'], pi['backend']] == ['rsi', 'pi']
    for arm in bundle['arms']:
        assert arm['usage'] == usage_totals(arm['requests'])
    assert rsi['usage']['total_tokens'] == 73685 and rsi['tokens_to_numerical_pass'] is None
    assert pi['usage']['total_tokens'] == 34512
    assert pi['tokens_to_numerical_pass']['total_tokens'] == 34512
    assert all(c['passed'] and c['max_abs_error'] == 0 for c in pi['validations'][-1]['cases'])
    assert hashlib.sha256((root / 'pi-kernel.cpp').read_bytes()).hexdigest() == pi['final_source_sha256']
    assert pi['tiling_review'] == 'passed'


def test_published_source_replay_never_calls_provider(tmp_path, monkeypatch):
    from benchmarks.pi_coding import validate_saved
    monkeypatch.setattr(validate_saved, 'LOCAL', tmp_path / 'local')
    source = tmp_path / 'source.cpp'
    source.write_text('device source')
    connection = tmp_path / 'connection.json'
    connection.write_text('{}')
    captured = []
    def validate(validator):
        captured.append(validator)
        assert (validator.root / 'kernel.cpp').read_bytes() == source.read_bytes()
        return {'status': 'passed', 'passed': True, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    monkeypatch.setattr(validate_saved.Validator, 'validate', validate)
    monkeypatch.setattr('benchmarks.pi_coding.gateway.requests.post', lambda *_args, **_kwargs: pytest.fail('no provider call'))
    assert validate_saved.main(['--source', str(source), '--connection', str(connection)]) == 0
    assert captured[0].task == 'matmul-add'
    with pytest.raises(SystemExit):
        validate_saved.main(['--source', str(source), '--connection', str(connection), '--expected-sha256', '0'*64])


def test_missing_pi_stops_before_board_or_provider(tmp_path, monkeypatch):
    from benchmarks.pi_coding import compare
    monkeypatch.setattr(compare, 'LOCAL', tmp_path)
    def unavailable():
        raise ValueError('Pi is not installed')
    monkeypatch.setattr(compare, 'runtime_preflight', unavailable)
    monkeypatch.setattr(compare.Validator, 'validate', lambda _: pytest.fail('no board call'))
    monkeypatch.setattr(compare, 'provider_preflight', lambda *_: pytest.fail('no paid call'))
    assert compare.main(['--budget-usd', '1', '--connection', str(tmp_path / 'absent.json')]) == 2
    report = json.loads(next(tmp_path.glob('comparisons/*/comparison.json')).read_text())
    assert report['status'] == 'blocked'
    assert report['error'] == 'Pi is not installed'
    assert report['budget_bounds_usd'] == {'rsi': 0, 'pi': 0}


def test_chained_reruns_use_original_allocation_and_refuse_overlap(tmp_path, monkeypatch):
    from benchmarks.pi_coding import compare
    from benchmarks.pi_coding.budget import Budget
    monkeypatch.setattr(compare, 'LOCAL', tmp_path)
    monkeypatch.setattr(compare, 'runtime_preflight', lambda: None)
    monkeypatch.setattr(compare.Validator, 'validate', lambda _: {'passed': True, 'status': 'passed'})
    monkeypatch.setattr(compare, 'provider_preflight', lambda *_: {'passed': False, 'error': {'message': 'offline stop'}})
    original = tmp_path / 'comparisons/original'
    original.mkdir(parents=True)
    (original / 'comparison.json').write_text(json.dumps({'status': 'completed_with_failures', 'total_limit_usd': 1}))
    for backend in ('rsi', 'pi'):
        Budget(original / (backend + '-budget.json'), .5).reserve({}, output_tokens=100, rates=(1, 1), label='prior')
    connection = tmp_path / 'connection.json'
    connection.write_text('{}')
    base = ['--connection', str(connection), '--no-knowledge']
    assert compare.main(base + ['--reuse-budget-from', 'original']) == 2
    second = next(p for p in (tmp_path / 'comparisons').iterdir() if p.name != 'original')
    with allocation_lock(original):
        with pytest.raises(SystemExit):
            compare.main(base + ['--reuse-budget-from', second.name])
    assert compare.main(base + ['--reuse-budget-from', second.name]) == 2
    for path in tmp_path.glob('comparisons/deepseek-*/comparison.json'):
        assert json.loads(path.read_text())['budget_ledger_directory'] == str(original.resolve())
        assert not (path.parent / 'rsi-budget.json').exists()
    assert len(Budget(original / 'rsi-budget.json', .5).snapshot()['requests']) == 1


def test_source_export_checks_hash_and_private_data(tmp_path, monkeypatch):
    from benchmarks.pi_coding import export_evidence
    monkeypatch.setattr(export_evidence, 'LOCAL', tmp_path)
    run = tmp_path / 'runs/pi'
    run.mkdir(parents=True)
    comparison = tmp_path / 'comparisons/pair'
    comparison.mkdir(parents=True)
    (comparison / 'comparison.json').write_text(json.dumps({'arms': [{'run': 'pi', 'backend': 'pi'}]}))
    source = b'// /home/private-user/workspace\n'
    (run / 'kernel.cpp').write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    result = {'backend': 'pi', 'status': 'passed', 'requests': [], 'final_source_sha256': digest,
              'validations': [{'passed': True, 'source_sha256': digest}]}
    (run / 'result.json').write_text(json.dumps(result))
    args = ['--comparison', 'pair', '--output', str(tmp_path / 'public'), '--include-passing-source']
    with pytest.raises(SystemExit):
        export_evidence.main(args)
    assert not (tmp_path / 'public').exists()
    (run / 'kernel.cpp').write_text('tampered')
    with pytest.raises(SystemExit):
        export_evidence.main(args)
    assert not (tmp_path / 'public').exists()
