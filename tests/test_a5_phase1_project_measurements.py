from dataclasses import replace
from statistics import median

import pytest

from benchmarks.a5kernels.phase1_curriculum import Capability
from benchmarks.a5kernels.phase1_project import ProjectGraph
from benchmarks.a5kernels.phase1_project_measurements import specialize_timing, timing_summary
from tests.test_a5_phase1_project_executor import _spec


def test_specialization_regenerates_graph_and_preserves_original():
    spec = _spec('movement')
    graph = ProjectGraph(tuple(replace(x, shape=(5,)) for x in spec.graph.inputs),
                         spec.graph.nodes, spec.graph.output)
    spec = replace(spec, shape=(5,), graph=graph,
                   capabilities=(*spec.capabilities, Capability.TIMING_SWEEP),
                   timing_lengths=(7, 17))
    variants = specialize_timing(spec)
    assert [x.shape for x in variants] == [(7,), (17,)]
    assert [x.graph.output_shape for x in variants] == [(7,), (17,)]
    assert spec.shape == (5,)


def test_aligned_sweep_variant_does_not_claim_padded_tail():
    spec = replace(_single_spec(), timing_lengths=(16, 17))
    aligned, padded = specialize_timing(spec)
    assert Capability.PADDED_TAIL not in aligned.capabilities
    assert Capability.PADDED_TAIL in padded.capabilities


def test_aligned_reduction_inputs_preserve_padded_output_capability():
    from benchmarks.a5kernels.phase1_project import GraphInput, GraphNode, ProjectOp, materialize_inputs, evaluate_reference
    spec = _single_spec()
    graph = ProjectGraph((GraphInput('x', (64,)),),
                         (GraphNode('out', ProjectOp.REDUCE_SUM, ('x',), 0),), 'out')
    spec = replace(spec, shape=(64,), graph=graph, padding_multiple=64,
                   capabilities=(Capability.SIMT_REDUCTION, Capability.PADDED_TAIL,
                                 Capability.TIMING_SWEEP), timing_lengths=(64, 128))
    for variant in specialize_timing(spec):
        assert Capability.PADDED_TAIL in variant.capabilities
        expected = evaluate_reference(variant, materialize_inputs(variant))
        assert expected.shape == (1,)
        assert expected.physical_shape == (64,)


def test_timing_bound_matches_executable_graph_dimensions():
    spec = replace(_single_spec(), timing_lengths=(64, 4096))
    assert specialize_timing(spec)[-1].shape == (4096,)
    with pytest.raises(ValueError, match=r'\[1, 4096\]'):
        replace(spec, timing_lengths=(64, 4097))


def test_multidimensional_sweep_is_actionable():
    spec = replace(_spec('movement'),
                   capabilities=(*_spec('movement').capabilities, Capability.TIMING_SWEEP),
                   timing_lengths=(7, 17))
    with pytest.raises(ValueError, match='one-dimensional'):
        specialize_timing(spec)


@pytest.mark.parametrize('metric_name', ['PIPE_UTILIZATION', 'MEMORY', 'PIPE_TIMELINE'])
def test_nonbasic_capture_requires_kernel_before_remote_dispatch(tmp_path, metric_name):
    from benchmarks.a5kernels.profiling import CampaignKind, CaptureCommand, ProfileMetric, ProfileRequest
    plan = ProjectBackend().measurement_plan(_candidate(tmp_path))
    request = ProfileRequest(plan, 'catlass-dsl', 'kernel', 3)
    metric = ProfileMetric[metric_name]
    with pytest.raises(ValueError, match=f'{metric.value} requires an exact kernel name'):
        CaptureCommand(CampaignKind.INTERMEDIATE, request, metric, 'replay')


def test_timing_summary_preserves_raw_values_and_labels():
    result = timing_summary([1., 3., 2.])
    assert result['samples_us'] == [1., 3., 2.]
    assert result['median_us'] == median([1., 3., 2.])
    assert result['dispersion_us'] > 0
    assert result['measurement'] == 'launch-plus-synchronize'


@pytest.mark.parametrize('values', [[1., 2.], [1., float('nan'), 3.], [1., -1., 3.]])
def test_invalid_samples_rejected(values):
    with pytest.raises(ValueError):
        timing_summary(values)


def _single_spec():
    spec = _spec('movement')
    graph = ProjectGraph(tuple(replace(x, shape=(5,)) for x in spec.graph.inputs),
                         spec.graph.nodes, spec.graph.output)
    return replace(spec, shape=(5,), graph=graph,
                   capabilities=(*spec.capabilities, Capability.TIMING_SWEEP),
                   timing_lengths=(7, 17))


class ProjectBackend:
    device = 3

    def __init__(self, *, mismatch=False):
        self.requests = []
        self.mismatch = mismatch

    def compile(self, request):
        from benchmarks.a5kernels.phase1_project_executor import CompileObservation
        self.requests.append(request)
        return CompileObservation(True, 'a' * 64, 'compiled', 'bundle')

    def run(self, request, compilation):
        from benchmarks.a5kernels.phase1_project_executor import RuntimeObservation
        values = request.expected.storage
        if self.mismatch:
            values = tuple(999. for _ in values)
        return RuntimeObservation(True, 'b' * 64, 'ran', values)

    def observation_payload(self, digest):
        return {'digest': digest}

    def measurement_plan(self, request):
        from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile
        return ExecutionPlan('request', 'measure-test', 'catlass-dsl',
                             (SourceFile('kernel.py', request.candidate_source),),
                             ('python', 'host.py'), (), ())


class ProfileBackend:
    def __init__(self):
        self.timing_commands = []
        self.captures = []

    def time_sample(self, command):
        from benchmarks.a5kernels.profiling import StudyTimingSample
        self.timing_commands.append(command)
        assert command.request.warm_up == 5
        assert command.request.launch_count == 20
        return StudyTimingSample(2., command.request.source_fingerprint,
                                 command.request.execution_id, command.replay_id, 'c' * 64)

    def timing_evidence_payload(self, replay):
        return {'replay': replay, 'stdout': 'A5KERNEL_TIMING_US=2.0'}

    def capture(self, command):
        from benchmarks.a5kernels.profiling import EvidenceArchive, ProfileCapture
        self.captures.append(command)
        return ProfileCapture(command.metric, command.request.source_fingerprint,
                              command.request.execution_id, command.replay_id,
                              ('project_kernel_exported',), (('duration', '2'),),
                              EvidenceArchive('compact.tar.gz', 'd' * 64, 1, '/raw', True, ()))

    def capture_evidence_payload(self, capture):
        return {'replay': capture.replay_id, 'raw_summary': list(capture.summary)}


def _candidate(tmp_path):
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    executor = ProjectExecutor(ProjectBackend(), EvidenceLedger(tmp_path / 'ledger'))
    return executor._request(executor.prepare(_single_spec()), '# source')


def test_sweep_verifies_regenerated_inputs_before_three_samples(tmp_path):
    from benchmarks.a5kernels.phase1_project_measurements import ProjectMeasurementBackend
    project, profiler = ProjectBackend(), ProfileBackend()
    result = ProjectMeasurementBackend(project, profiler).timing(_candidate(tmp_path))
    assert [request.inputs[0].shape for request in project.requests] == [(7,), (17,)]
    assert len(profiler.timing_commands) == 6
    assert [row['shape'] for row in result['summary']['timing']] == [[7], [17]]
    assert result['artifact_payloads']


def test_host_mismatch_prevents_measurement(tmp_path):
    from benchmarks.a5kernels.phase1_project_measurements import ProjectMeasurementBackend
    profiler = ProfileBackend()
    with pytest.raises(ValueError, match='host correctness'):
        ProjectMeasurementBackend(ProjectBackend(mismatch=True), profiler).timing(_candidate(tmp_path))
    assert not profiler.timing_commands


@pytest.mark.parametrize('metric,group', [('pipe-utilization', 'PipeUtilization'),
                                         ('memory-throughput', 'Memory'),
                                         ('timeline', 'PipeTimeline')])
def test_profile_discovers_export_then_binds_selected_metric(tmp_path, metric, group):
    from benchmarks.a5kernels.phase1_project_measurements import ProjectMeasurementBackend
    profiler = ProfileBackend()
    result = ProjectMeasurementBackend(ProjectBackend(), profiler).profile(_candidate(tmp_path), metric)
    basic, capture = profiler.captures
    assert basic.metric.value == 'BasicInfo'
    assert basic.kernel_name is None
    assert capture.metric.value == group
    assert capture.kernel_name == 'project_kernel_exported'
    assert result['summary']['duration_semantics'].startswith('instrumented')


def test_ambiguous_basicinfo_is_actionable_without_guessing(tmp_path):
    from benchmarks.a5kernels.phase1_project_measurements import ProjectMeasurementBackend
    class Ambiguous(ProfileBackend):
        def capture(self, command):
            return replace(super().capture(command), exported_kernels=('first', 'second'))
    profiler = Ambiguous()
    with pytest.raises(ValueError, match='exactly one exported'):
        ProjectMeasurementBackend(ProjectBackend(), profiler).profile(_candidate(tmp_path), 'timeline')
    assert len(profiler.captures) == 1


def test_concrete_backend_raw_evidence_is_reconstructible(tmp_path):
    import base64
    import hashlib
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    from benchmarks.a5kernels.profiling import EvidenceArchive, ProfileCapture, ProfileMetric
    backend = BZProfileBackend(validation_wrapper='/profiles/catlass-validation.sh',
                               collection_wrapper='/skills/collect_profile.sh',
                               catlass_source='/catlass', evidence_directory=str(tmp_path))
    directory = tmp_path / 'replay'
    directory.mkdir()
    data = b'compact archive bytes'
    (directory / 'compact.tar.gz').write_bytes(data)
    capture = ProfileCapture(ProfileMetric.MEMORY, 'a' * 64, 'b' * 64, 'replay',
                             ('kernel',), (('Memory', '3'),),
                             EvidenceArchive('compact.tar.gz', hashlib.sha256(data).hexdigest(),
                                             len(data), '/retained/tree', True, ()))
    payload = backend.capture_evidence_payload(capture)
    assert base64.b64decode(payload['archive']) == data
    assert payload['capture']['evidence']['remote_tree'] == '/retained/tree'
    (directory / 'compact.tar.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        backend.capture_evidence_payload(capture)


def test_timeline_reader_uses_normalized_summary(tmp_path):
    import io
    import tarfile
    from benchmarks.a5kernels.profiling import ProfileMetric
    from benchmarks.a5kernels.profiling_bz import _read_metric_archive
    path = tmp_path / 'timeline.tar.gz'
    data = b'{"pipe_timeline":{"events":3,"pipes":["vector","mte2"]}}'
    with tarfile.open(path, 'w:gz') as archive:
        member = tarfile.TarInfo('derived/timeline-summary.json')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    kernels, summary = _read_metric_archive(path, ProfileMetric.PIPE_TIMELINE)
    assert kernels == ()
    assert summary[0][0] == 'timeline-summary'
    assert 'vector' in summary[0][1]


def test_timeline_normalization_recovers_same_retained_operation(tmp_path):
    import subprocess
    from benchmarks.a5kernels.profiling import CampaignKind, CaptureCommand, ProfileMetric, ProfileRequest
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == 'result':
            return subprocess.CompletedProcess(argv, 0, 'completed', '')
        if 'logs' in argv:
            return subprocess.CompletedProcess(argv, 0, 'A5TIMELINE_NORMALIZED=1\n', '')
        if '--operation' in argv:
            return subprocess.CompletedProcess(argv, 75, '', 'observation interrupted')
        return subprocess.CompletedProcess(argv, 0, '', '')
    backend = BZProfileBackend(validation_wrapper='/profiles/catlass-validation.sh',
                               collection_wrapper='/skills/scripts/collect_profile.sh',
                               catlass_source='/catlass', evidence_directory=str(tmp_path),
                               process_runner=runner)
    plan = ProjectBackend().measurement_plan(_candidate(tmp_path))
    request = ProfileRequest(plan, 'catlass-dsl', 'kernel', 3)
    backend._normalize_timeline(CaptureCommand(CampaignKind.INTERMEDIATE, request,
                                              ProfileMetric.PIPE_TIMELINE, 'replay', 'kernel'), '/raw')
    launch = next(x for x in calls if '--operation' in x)
    operation = launch[launch.index('--operation') + 1]
    logs = next(x for x in calls if 'logs' in x)
    assert logs[logs.index('--name') + 1] == operation
    assert len([x for x in calls if '--operation' in x]) == 1


def test_adaptive_timing_masks_physical_device_but_profile_is_direct(tmp_path):
    from benchmarks.a5kernels.profiling import ProfileRequest
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    backend = BZProfileBackend(validation_wrapper='/profiles/catlass-validation.sh',
                               collection_wrapper='/skills/collect_profile.sh',
                               catlass_source='/catlass', evidence_directory=str(tmp_path))
    backend._bound_workload_argv = lambda request: ('python', 'host.py')
    plan = ProjectBackend().measurement_plan(_candidate(tmp_path))
    plan = replace(plan, environment=plan.environment.with_binding('A5PROJECT_MASKED_TIMING', '1'))
    request = ProfileRequest(plan, 'catlass-dsl', 'kernel', 3)
    timing = backend._adapter_argv('timing', 'run', request)
    prefix = timing[:timing.index('--')]
    assert prefix[prefix.index('--device') + 1] == '3'
    assert 'A5KERNEL_PROFILE_DIRECT_PHYSICAL_DEVICE=1' not in timing
    profile = backend._adapter_argv('profile', 'profile', request)
    assert '--device' not in profile[:profile.index('--')]
    assert 'A5KERNEL_PROFILE_DIRECT_PHYSICAL_DEVICE=1' in profile


def test_native_vendor_trace_is_discovered_and_actually_normalized(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    from benchmarks.a5kernels.profiling import CampaignKind, CaptureCommand, ProfileMetric, ProfileRequest
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    raw = tmp_path / 'PROF_00001' / 'device_4' / 'output'
    raw.mkdir(parents=True)
    (raw / 'trace.json').write_text(json.dumps({'traceEvents': [
        {'ph': 'X', 'ts': 1, 'dur': 4, 'cat': 'vector', 'name': 'vec'},
        {'ph': 'X', 'ts': 2, 'dur': 3, 'cat': 'mte2', 'name': 'copy'},
    ]}))
    helper = Path.home() / '.agents/skills/ascend-profiling/scripts/summarize_timelines.py'
    if not helper.is_file():
        pytest.skip('user-wide ascend-profiling skill is not installed on this test host')
    # Execute the actual discovery payload against a vendor-shaped tree; only
    # transport staging is replaced. This catches filename-contract regressions.
    def runner(argv, **kwargs):
        if '--operation' not in argv:
            return subprocess.CompletedProcess(argv, 0, '', '')
        index = argv.index('--') + 1
        payload = list(argv[index:])
        payload[-1] = str(helper)
        return subprocess.run(payload, **kwargs)
    backend = BZProfileBackend(validation_wrapper='/profiles/catlass-validation.sh',
                               collection_wrapper='/skills/scripts/collect_profile.sh',
                               catlass_source='/catlass', evidence_directory=str(tmp_path),
                               process_runner=runner)
    plan = ProjectBackend().measurement_plan(_candidate(tmp_path))
    request = ProfileRequest(plan, 'catlass-dsl', 'kernel', 3)
    backend._normalize_timeline(CaptureCommand(CampaignKind.INTERMEDIATE, request,
                                              ProfileMetric.PIPE_TIMELINE, 'native', 'kernel'),
                                str(tmp_path / 'PROF_00001'))
    summary = json.loads((tmp_path / 'PROF_00001' / 'adaptive-timeline' /
                          'timeline-summary.json').read_text())
    assert summary['pipe_timeline']['pipes'] == ['mte2', 'vector']
    assert summary['pipe_timeline']['overlap_duration']['mte2:vector'] == 3


def test_profile_driver_initialization_and_conversion_do_not_launch_helper_kernels():
    import ast
    from benchmarks.a5kernels.phase1_project_bz import _PROJECT_DRIVER
    calls = []
    class Tensor:
        device = 'cpu'
        def to(self, device):
            if device not in ('cpu', 'npu'):
                assert self.device == 'cpu', 'device Cast would pollute BasicInfo'
                calls.append(('cast', self.device))
                return self
            calls.append(('transfer', device))
            self.device = device
            return self
        def cpu(self):
            return self.to('cpu')
        def float(self):
            assert self.device == 'cpu', 'device Cast would pollute BasicInfo'
            calls.append(('cast', self.device))
            return self
        def reshape(self, shape):
            return self
        def tolist(self):
            return [1.]
    class Torch:
        float32 = "float32"
        def full(self, shape, value, *, dtype, device='cpu'):
            assert device == 'cpu', 'device FillD would pollute BasicInfo'
            calls.append(('fill', device))
            return Tensor()
    namespace = {'torch': Torch(), 'output_shape': (1,), 'output_dtype': 'float16',
                 'low_precision': False}
    main = next(node for node in ast.parse(_PROJECT_DRIVER).body
                if isinstance(node, ast.FunctionDef) and node.name == '_main')
    blocks = [main, *(node for node in main.body if isinstance(node, ast.Try))]
    assignments = [node for block in blocks for node in block.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)]
    # Execute every output initialization step in order, then CPU observation.
    # This catches accidental device Fill/Cast helpers without relying on one
    # specific spelling of the driver's intermediate tensors.
    for name in ('output', 'host_output', 'values'):
        for node in assignments:
            if node.targets[0].id != name:
                continue
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, '<project-driver>', 'exec'), namespace)
    assert calls == [('fill', 'cpu'), ('cast', 'cpu'), ('transfer', 'npu'),
                     ('transfer', 'cpu'), ('cast', 'cpu')]


def test_current_matrix_shape_timing_needs_no_sweep():
    spec = _spec("movement")
    assert len(spec.shape) > 1
    assert specialize_timing(spec) == (spec,)


def test_device_duration_requires_units_and_preserves_overlapping_tasks():
    from types import SimpleNamespace
    from benchmarks.a5kernels.phase1_project_measurements import device_duration_summary
    capture = SimpleNamespace(exported_kernels=("kernel",), summary=(
        ("OpBasicInfo.csv.gz:Task Duration(us)", "7.2"),
        ("OpBasicInfo.csv.gz:Duration(us)", "5.1"),
        ("duration", "999")))
    result = device_duration_summary(capture)
    assert result["task_durations_us"] == [7.2, 5.1]
    assert result["instrumented"] is True
    capture.summary = (("duration", "999"),)
    assert device_duration_summary(capture)["status"] == "unavailable"
