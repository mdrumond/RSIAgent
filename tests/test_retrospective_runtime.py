import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend import retrospective_runtime as runtime
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from test_a5_phase1_adaptive import _project


def request(tmp_path, seconds=30):
    spec = ProjectSpec.from_mapping(_project('p', 'gm-ub-movement'))
    inputs = materialize_inputs(spec)
    return AdaptiveProjectRequest(spec, 1, b'{}', tmp_path, inputs, evaluate_reference(spec, inputs),
                                  remaining_seconds=seconds)


def compose(tmp_path, monkeypatch, *, stderr='', now=None, stopped=False):
    calls = []
    native = dict(target='a3-product', arch='dav-2201', execution_profile='gz-a3',
                  transport_target='registry-target', runtime='named-runtime', registry_workspace='/owned')
    monkeypatch.setattr(runtime, 'load_native_config', lambda *a: native)
    class Transport:
        def __init__(self, **options):
            self.options = options
            self.remaining_seconds = options['remaining_seconds']
            self.runtime_identity = {'runtime': options['runtime']}
        def compile(self, plan, root):
            calls.append('probe-compile')
            if now is not None:
                now[0] += 4
            return SimpleNamespace(plan=plan, library_sha256='b'*64, stdout='', stderr=stderr)
        def execute_receipt(self, compilation):
            calls.append('probe-execute')
            return SimpleNamespace(job_handle='remote:registry-target:job:probe', exit_code=0,
                output=tuple(sum(xs) for xs in zip(*(t.values for t in compilation.plan.inputs))))
    class Actor:
        immutable_context = b'original'
        query_knowledge = None
        executor = SimpleNamespace(execute=lambda *a, **k: calls.append('extra-check'))
        def __call__(self, value):
            calls.append(('actor', value.remaining_seconds))
            return value
    def production(settings, **options):
        transport = options['native_options']['transport_factory'](cpl_remote='/skill/cpl-remote',
            validation_wrapper='must-not-run', remote_workspace='/owned/arm', physical_device=3,
            profile='gz-a3', state_directory=tmp_path/'transport')
        paths = SimpleNamespace(root=tmp_path, **{k: native[k] for k in ('target','arch','execution_profile')})
        return dict(actor=Actor(), paths=paths, transport=transport)
    monkeypatch.setattr(runtime, 'CplNativeCandidateBackend', Transport)
    monkeypatch.setattr(runtime, 'compose_production', production)
    if now is not None:
        monkeypatch.setattr(runtime.time, 'monotonic', lambda: now[0])
    stop = tmp_path/'STOP'
    if stopped:
        stop.touch()
    bound = runtime.compose_cpl_production({'phase1_arguments':['--config','fixture'], 'stop_file':str(stop)},
        target='a3', root=tmp_path, initial_memory=b'', max_seconds=100)
    return bound, calls


def test_gate_precedes_actor_charges_elapsed_and_preserves_configured_route(tmp_path, monkeypatch):
    now = [0.0]
    bound, calls = compose(tmp_path, monkeypatch, now=now)
    result = bound['actor'](request(tmp_path))
    assert calls == ['probe-compile', 'probe-execute', ('actor', 26)]
    assert result.remaining_seconds == 26
    options = bound['transport'].options
    assert options['target'] == 'registry-target' and options['runtime'] == 'named-runtime'
    assert options['physical_device'] == 3 and 'validation_wrapper' not in options
    bound['actor'].query_knowledge = 'frozen snapshot query'
    assert bound['actor']._wrapped.query_knowledge == 'frozen snapshot query'
    bound['actor'](request(tmp_path))
    assert calls[-1][0] == 'actor' and calls.count('probe-compile') == 1


def test_shell_setup_failure_blocks_actor_and_retains_report(tmp_path, monkeypatch):
    bound, calls = compose(tmp_path, monkeypatch, stderr='sh: 28: source: not found\n')
    with pytest.raises(ProjectInfrastructureError):
        bound['actor'](request(tmp_path))
    assert calls == ['probe-compile']
    report = json.loads(next(tmp_path.glob('environment-checks/**/report.json')).read_text())
    assert report['status'] == 'failed'
    assert report['workers'][0]['compile_stderr']


def test_stop_blocks_dispatch_and_extra_verifier_checks_are_gated(tmp_path, monkeypatch):
    bound, calls = compose(tmp_path, monkeypatch, stopped=True)
    with pytest.raises(TimeoutError):
        bound['actor'](request(tmp_path))
    assert not calls
    (tmp_path/'STOP').unlink()
    bound['actor'].executor.execute('variant')
    assert calls == ['probe-compile', 'probe-execute', 'extra-check']
    (tmp_path/'STOP').touch()
    with pytest.raises(TimeoutError):
        bound['actor'].executor.execute('another variant')
    assert calls[-1] == 'extra-check'


def test_native_identity_reports_the_selected_registered_runtime():
    from benchmarks.ascend.production import native_target_runtime
    transport = SimpleNamespace(_workspace='/owned', _device=3, profile='gz-a3', runtime='py312-torch')
    config = SimpleNamespace(target='a3-product', arch='dav-2201', compiler_version='pinned',
                             environment_identity={'cann_version':'pinned-cann'})
    identity = native_target_runtime(config, transport)
    assert identity['runtime'] == 'py312-torch' and identity['device'] == '3'
