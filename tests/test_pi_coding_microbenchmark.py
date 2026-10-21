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
