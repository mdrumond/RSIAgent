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


def test_multidimensional_sweep_is_actionable():
    spec = replace(_spec('movement'),
                   capabilities=(*_spec('movement').capabilities, Capability.TIMING_SWEEP),
                   timing_lengths=(7, 17))
    with pytest.raises(ValueError, match='one-dimensional'):
        specialize_timing(spec)


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
