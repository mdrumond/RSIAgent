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


def compose(tmp_path, monkeypatch, *, stderr='', now=None, stopped=False, device=3, operation_root=None):
    calls = []
    native = dict(target='a3_product', arch='dav-2201', execution_profile='gz-a3',
                  transport_target='registry-target', runtime='named-runtime', registry_workspace='/owned', health_workspace='/owned/incoming',
                  compiler_version='pinned', environment_identity={'torch':'torch-v','torch_npu':'npu-v','cann_version':'runtime: cann-v','soc_version':255})
    monkeypatch.setattr(runtime, 'load_native_config', lambda *a: native)
    class Transport:
        def __init__(self, **options):
            self.options = options
            self.remaining_seconds = options['remaining_seconds']
            self.runtime_identity = dict(target=options['target'], runtime=options['runtime'], physical_device=options['physical_device'])
        def recover_owned_operations(self, **kwargs): pass
        def run_remote(self, *args, **kwargs):
            return SimpleNamespace(returncode=0, job_handle='remote:registry-target:job:health', stdout=json.dumps({
                'queries':{kind:{'exit':0, 'text':text} for kind,text in {
                    'health':'Health Status : OK', 'ecc':'Errors : 0', 'memory':'HBM',
                    'usages':'HBM Usage Rate(%) : 0', 'proc-mem':'No process in device', 'board':'Chip Name : a3\nNPU Name : product'}.items()},
                'compiler':'pinned', 'versions':{'torch':'torch-v','torch-npu':'npu-v'}, 'toolkit':'runtime: cann-v','soc_version':255,'arch':'dav-2201'}))
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
        request_timing = staticmethod(lambda *a, **k: calls.append('timing') or {'receipt':'timing'})
        request_profile = staticmethod(lambda *a, **k: calls.append('profile') or {'receipt':'profile'})
        executor = SimpleNamespace(execute=lambda *a, **k: calls.append('extra-check'))
        def __call__(self, value):
            calls.append(('actor', value.remaining_seconds))
            return value
    def production(settings, **options):
        transport = options['native_options']['transport_factory'](cpl_remote='/skill/cpl-remote',
            validation_wrapper='must-not-run', remote_workspace='/owned/arm', physical_device=device,
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
    bound = runtime.compose_cpl_production({'phase1_arguments':['--config','fixture'], 'stop_file':str(stop),
         'native_operation_root':str(operation_root) if operation_root else None},
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


@pytest.mark.parametrize('worker, expected', [
    ({'target':'another-registry-target'}, 'another-registry-target'),
    ({'target':'a3-product','transport_target':'explicit-route'}, 'explicit-route'),
    ({'target':'a3-product'}, 'registry-target'),
])
def test_native_worker_merge_distinguishes_product_from_route(tmp_path, worker, expected):
    from benchmarks.ascend.execution import apply_worker_config
    path = tmp_path/'worker.json'
    path.write_text(json.dumps(worker))
    merged = apply_worker_config({'target':'a3-product','transport_target':'registry-target'}, path)
    assert merged['target'] == 'a3-product' and merged['transport_target'] == expected


def test_full_native_worker_arm_copies_keep_registered_route_and_isolation(tmp_path, monkeypatch):
    from benchmarks.ascend import subject_runtime
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    native = dict(target='a3-product', transport_target='registry-target', runtime='py312-torch',
        arch='dav-2201', execution_profile='gz-a3', physical_device=3,
        guide='retained-guide', cpl_skills_revision='a'*40, cpl_remote='/skill/cpl-remote',
        validation_wrapper='/unused', remote_workspace='/owned/run',
        environment_identity={'cann_version':'pinned','profiler_version':'unavailable'})
    path = tmp_path/'native.json'; path.write_text(json.dumps(native))
    routes = []
    def compose(settings, **options):
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config'); parser.add_argument('--worker-config')
        arguments, _ = parser.parse_known_args(settings['phase1_arguments'])
        merged = runtime.load_native_config(arguments.config, arguments.worker_config)
        routes.append(merged)
        if len(routes) > 1:
            assert settings['environment_check_cache'] == str(tmp_path/'shared-probes')
            assert settings['native_operation_root'] == str(tmp_path/'campaign')
        return dict(bound(options['root']), environment_check_cache=str(tmp_path/'shared-probes'),
                    native_operation_root=str(tmp_path/'campaign'))
    class Execution:
        def __init__(self, *args): pass
        def graph(self, request, **kwargs): return request
    monkeypatch.setattr(subject_runtime, 'SubjectExecutor', Execution)
    campaign = subject_runtime.build_subject_campaign({'phase1_arguments':['--config',str(path)]},
        target='a3',root=tmp_path/'campaign',subjects=[Subject('s','copies')],initial_memory=b'',compose=compose)
    item = dict(project_id='transfer', spec=_project('transfer','gm-ub-movement'))
    for arm in ['control','treatment']:
        campaign.execute_trial_arm(item,campaign.memory.snapshot(),tmp_path/arm,remaining_seconds=60)
    assert [row['transport_target'] for row in routes] == ['registry-target']*3
    assert [row['target'] for row in routes] == ['a3-product']*3
    assert routes[1]['remote_workspace'] != routes[2]['remote_workspace']
    assert routes[1]['runtime'] == routes[2]['runtime'] == 'py312-torch'


@pytest.mark.parametrize('error_name, expected', [('PiCancelled','operator-stop'),('PiDeadlineError','operator-stop'),('PiProviderError',None)])
def test_actual_stop_marks_cancellation_but_preserves_provider_failure(tmp_path,monkeypatch,error_name,expected):
    from coding.pi import rpc
    bound,_=compose(tmp_path,monkeypatch)
    actor=bound['actor']
    error=getattr(rpc,error_name)('retained error')
    def fail(self,request):
        (tmp_path/'STOP').touch()
        raise error
    monkeypatch.setattr(type(actor._wrapped),'__call__',fail)
    with pytest.raises(type(error)):
        actor(request(tmp_path))
    assert getattr(error,'interruption_reason',None)==expected


def test_changed_device_has_distinct_retained_namespace_and_preserves_old_records(tmp_path, monkeypatch):
    first, _ = compose(tmp_path, monkeypatch, device=1)
    original = first['transport'].options['state_directory']
    original.mkdir(parents=True)
    journal = original/'operation.json'
    journal.write_text('retained original allocation')
    second, _ = compose(tmp_path, monkeypatch, device=2)
    assert original != second['transport'].options['state_directory']
    assert journal.read_text() == 'retained original allocation'
    same, _ = compose(tmp_path, monkeypatch, device=1)
    assert same['transport'].options['state_directory'] == original


def test_background_trial_pending_operation_blocks_foreground_until_settled(tmp_path, monkeypatch):
    binding = tmp_path/'foreground'
    bound, calls = compose(binding, monkeypatch, operation_root=tmp_path)
    journal = tmp_path/'memory-reviews/trial/control/runtime/transport/operation.json'
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps(dict(state='queued', handle='remote:registry-target:job:trial')))
    with pytest.raises(ProjectInfrastructureError, match='unsettled'):
        bound['actor'](request(binding))
    assert calls == []
    assert json.loads(journal.read_text())['state'] == 'queued'
    journal.write_text(json.dumps(dict(state='completed', handle='remote:registry-target:job:trial')))
    bound['actor'](request(binding))
    assert calls[:2] == ['probe-compile','probe-execute'] and calls[-1][0] == 'actor'
    assert bound['native_operation_root'] == str(tmp_path.resolve())


@pytest.mark.parametrize('name', ['request_timing','request_profile'])
def test_measurement_callbacks_readmit_after_last_correctness_run(tmp_path, monkeypatch, name):
    bound, calls = compose(tmp_path/'foreground', monkeypatch, operation_root=tmp_path)
    bound['actor'].executor.execute('correctness')
    before = list(calls)
    pending = tmp_path/'memory-reviews/trial/runtime/operation.json'
    pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps(dict(state='queued',handle='remote:registry-target:job:trial')))
    callback = getattr(bound['actor'],name)
    with pytest.raises(ProjectInfrastructureError, match='unsettled'):
        callback({'replay_id':'same'},None)
    assert calls == before
    pending.write_text(json.dumps(dict(state='completed',handle='remote:registry-target:job:trial')))
    first = callback({'replay_id':'same'},None)
    assert callback({'replay_id':'same'},None) == first
    assert calls.count(name.removeprefix('request_')) == 2


@pytest.mark.parametrize('bound_case', [False, True])
def test_campaign_reservation_serializes_parent_and_trial_admission_through_dispatch(tmp_path, monkeypatch, bound_case):
    import threading
    from contextlib import contextmanager
    parent, parent_calls = compose(tmp_path/'parent',monkeypatch,operation_root=tmp_path)
    trial, trial_calls = compose(tmp_path/'trial/runtime',monkeypatch,operation_root=tmp_path)
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def execute(*args):
        entered.set()
        assert release.wait(5)
        parent_calls.append('device-settled')
    parent['actor'].executor._executor.execute = execute
    if bound_case:
        parent['actor'].executor._executor.bind_case = lambda *args: SimpleNamespace(execute=execute)
        trial_execute = trial['actor'].executor._executor.execute
        trial['actor'].executor._executor.bind_case = lambda *args: SimpleNamespace(execute=trial_execute)
        parent['actor'].executor = parent['actor'].executor.bind_case(None, ())
        trial['actor'].executor = trial['actor'].executor.bind_case(None, ())
    reserve = trial['actor'].executor._check.reserve
    @contextmanager
    def observed_reserve():
        attempted.set()
        with reserve(): yield
    trial['actor'].executor._check.reserve = observed_reserve
    def run(bound):
        try: bound['actor'].executor.execute('candidate')
        except BaseException as error: errors.append(error)
    first, second = threading.Thread(target=run,args=(parent,)), threading.Thread(target=run,args=(trial,))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert attempted.wait(5)
        assert not trial_calls  # Neither probe nor candidate dispatch raced the parent.
    finally:
        release.set()
        first.join(5)
        if second.ident is not None: second.join(5)
    assert not first.is_alive() and not second.is_alive() and not errors
    assert parent_calls[-1] == 'device-settled'
    assert trial_calls == ['probe-compile','probe-execute','extra-check']


@pytest.mark.parametrize('operation', ['compile','execute'])
@pytest.mark.parametrize('case_offset', [1,2])
def test_bound_original_or_remedial_case_keeps_admission_and_exact_inputs(tmp_path, monkeypatch, operation, case_offset):
    bound, calls = compose(tmp_path/'foreground',monkeypatch,operation_root=tmp_path)
    spec = request(tmp_path).spec
    inputs = tuple(materialize_inputs(spec))
    dispatched = []
    class Executor:
        def __init__(self, retained=None): self.retained = retained
        def bind_case(self, selected, values):
            assert selected == spec
            return Executor(values)
        def compile(self, selected, source):
            dispatched.append(('compile',self.retained))
            return self.retained
        def execute(self, selected, source):
            dispatched.append(('execute',self.retained))
            return self.retained
    bound['actor'].executor._executor = Executor()
    selected = bound['actor'].executor.bind_case(spec,inputs)
    if case_offset == 2:
        selected = selected.bind_case(spec,inputs)
    pending = tmp_path/'trial/runtime/operation.json'
    pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps(dict(state='running',handle='remote:registry-target:job:trial')))
    with pytest.raises(ProjectInfrastructureError,match='unsettled'):
        getattr(selected,operation)(spec,'candidate')
    assert not dispatched and not calls
    pending.write_text(json.dumps(dict(state='completed',handle='remote:registry-target:job:trial')))
    assert getattr(selected,operation)(spec,'candidate') == inputs
    assert dispatched == [(operation,inputs)]
