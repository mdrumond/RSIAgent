import ast
import hashlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optimizer.measurement import aggregate_rows, parse_capture, capture


def test_pinned_cannbot_differential():
    source = Path(__file__).parents[1] / 'optimizer/vendor/cannbot_aggregate.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    exact_function = ast.get_source_segment(source.read_text(), function) + '\n'
    assert hashlib.sha256(exact_function.encode()).hexdigest() == (
        'd269af7cffead124102b319723fd8921ed4b521be894e1f51f047510b313fa17')
    namespace = {'np': np, 'logger': logging.getLogger(), 'Tuple': tuple,
                 'BAD_ROW_K': 50, 'BAD_ROW_FLOOR_US': 10.0}
    exec(compile(tree, str(source), 'exec'), namespace)
    rng = np.random.default_rng(9)
    for repeats in (1, 5, 50):
        for count in (1, 3, 50, 100, 151):
            values = rng.uniform(.1, 3, count).tolist() + [1000.0]
            expected = namespace['_aggregate_kernel_rows']('x', values, repeats)
            result = aggregate_rows('x', values, repeats)
            assert (result['mean_us'], result['launch_count']) == pytest.approx(expected)


def test_multi_launch_and_rejected_samples(tmp_path):
    report = tmp_path / 'kernel_details.csv'
    report.write_text('Name,Duration(us)\na,2\na,2\na,2\na,1000\nb,3\nb,3\n')
    result = parse_capture(tmp_path, repeats=2)
    assert result['latency_us'] == 5
    assert result['operators'][0]['removed_indices'] == [3]
    assert result['operators'][0]['discarded_prefix_count'] == 1
    assert len(result['operators'][0]['raw_durations_us']) == 4


@pytest.mark.parametrize('duration', ['nan', 'inf', '-1', 'oops'])
def test_bad_duration_fails_instead_of_fabricating_latency(tmp_path, duration):
    (tmp_path / 'kernel_details.csv').write_text('Name,Duration(us)\na,' + duration)
    with pytest.raises(ValueError):
        parse_capture(tmp_path, repeats=1)


def test_exact_capture_schedule_and_boundary(tmp_path, monkeypatch):
    calls, options = [], {}
    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def step(self):
            calls.append('step')
    def make_profile(**kwargs):
        options.update(kwargs)
        return Context()
    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw, AiCMetrics=SimpleNamespace(PipeUtilization=1),
        ProfilerLevel=SimpleNamespace(Level1=1), ProfilerActivity=SimpleNamespace(NPU=1),
        profile=make_profile, schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: path)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=profiler))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: calls.append('sync'))))
    capture(lambda: calls.append('invoke'), tmp_path / 'raw', 5, 50)
    assert calls == ['invoke', 'sync', 'step'] * 55
    assert options['schedule'] == dict(wait=0, warmup=5, active=50, repeat=1)
    assert options['experimental_config'] == dict(aic_metrics=1, profiler_level=1,
                                                 l2_cache=False, data_simplification=False)


def test_legacy_task_time_schema(tmp_path):
    (tmp_path / 'task_time_0.csv').write_text('kernel_name,task_time(us)\nx,3\nx,5\n')
    assert parse_capture(tmp_path, repeats=2)['latency_us'] == 4
