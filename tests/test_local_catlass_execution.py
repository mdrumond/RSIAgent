"""Local payload execution retains the existing host oracle and profiling inputs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import benchmarks.ascend.local_catlass as local
from benchmarks.a5kernels.phase1_project_executor import CompileObservation
from tests.test_a5_phase1_project_bz import _request


def setup_executor(monkeypatch, tmp_path, response):
    calls = []
    class Executor:
        runtime_provenance = (('catlass_revision', 'a' * 40), ('execution_profile', 'local'))
        arguments = ('a' * 40, 'b' * 64, 'c' * 64, 'd' * 40, 'd' * 40, 'actual-cann')
        def __init__(self, config):
            self.root = tmp_path / 'workspace'
        def execute(self, argv, operation, **kwargs):
            operation.mkdir(parents=True, exist_ok=True)
            (operation / 'transport.json').write_text(json.dumps({'handle': 'local:real-operation'}))
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, response, '')
    monkeypatch.setattr(local, 'LocalCatlassExecutor', Executor)
    return calls


def test_local_project_reuses_exact_bundle_and_reports_observed_values(tmp_path, monkeypatch):
    calls = setup_executor(monkeypatch, tmp_path,
        'A5PROJECT_RESULT=' + json.dumps({'status': 'compiled', 'diagnostic': 'compiled exact source'}))
    config = dict(workspace=str(tmp_path / 'workspace'), physical_device=2, arch='actual-arch')
    backend = local.LocalCatlassProjectBackend(config, evidence_root=tmp_path / 'evidence')
    request = _request(tmp_path, 'movement')
    compiled = backend.compile(request)
    assert compiled.passed
    assert calls[0][0][0] == sys.executable
    assert Path(calls[0][0][3]).read_text() == request.candidate_source
    evidence = backend.observation_payload(compiled.evidence_sha256)
    assert evidence['handle'].startswith('local:')
    assert evidence['execution_identity']['physical_device'] == 2
    assert not any('session.sh' in arg or 'cpl-remote' in arg for arg in calls[0][0])


def test_local_project_rejects_missing_driver_receipt(tmp_path, monkeypatch):
    setup_executor(monkeypatch, tmp_path, 'unstructured output')
    backend = local.LocalCatlassProjectBackend(
        dict(workspace=str(tmp_path), physical_device=0, arch='actual-arch'), evidence_root=tmp_path / 'evidence')
    with pytest.raises(RuntimeError, match='driver failed'):
        backend.compile(_request(tmp_path, 'movement'))


def test_local_profile_stages_project_without_changing_host_kernel_payload(tmp_path, monkeypatch):
    setup_executor(monkeypatch, tmp_path, '')
    config = dict(workspace=str(tmp_path / 'workspace'), physical_device=0, arch='actual-arch')
    backend = local.LocalCatlassProjectBackend(config, evidence_root=tmp_path / 'evidence')
    request = _request(tmp_path, 'movement')
    plan = backend.measurement_plan(request)
    from benchmarks.a5kernels.profiling import ProfileRequest
    profile = ProfileRequest(plan, 'catlass', '*', 0)
    measurements = local.LocalProfileBackend(config, evidence_directory=tmp_path / 'measurements')
    argv = measurements._workload(profile)
    assert argv[-6:] == list(measurements._executor.arguments)
    assert (measurements._executor.root / argv[3]).read_text() == request.candidate_source
