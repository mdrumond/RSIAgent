"""CANNBot-compatible device-operator timing; preparation belongs to the caller."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

CANNBOT_SHA256 = 'cf33f091aa645731cf5a47ade64434b3c60b55ae75591c9226e1e769fed5707a'
POLICY = 'cannbot-median50-floor10-trim-prefix-summed-operators-v1'


def aggregate_rows(name, durations, repeats):
    """Retain the pinned CANNBot policy, including its prefix trimming behavior."""
    values = [float(v) for v in durations]
    if repeats < 1 or not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('positive repeats and finite nonnegative duration rows required')
    launches = round(len(values) / repeats)
    threshold = max(median(values) * 50, 10.0)
    removed = [i for i, value in enumerate(values) if value > threshold]
    clean = [value for value in values if value <= threshold]
    retained = len(clean) - len(clean) % repeats
    prefix = len(clean) - retained if retained >= repeats else 0
    mean = sum(clean[-retained:]) / repeats if retained >= repeats else sum(clean) / repeats
    return {'name': name, 'mean_us': mean if launches else 0.0,
            'launch_count': launches, 'launch_count_exact': len(values) / repeats,
            'raw_durations_us': values, 'removed_indices': removed,
            'outlier_threshold_us': threshold, 'discarded_prefix_count': prefix}


def file_identity(path):
    path = Path(path)
    return {'path': str(path), 'size': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_capture(root, repeats=50):
    """Read one capture schema, never sum duplicate exports of the same capture."""
    root = Path(root)
    schemas = [('kernel_details.csv', 'Name', 'Duration(us)'),
               ('op_summary*.csv', 'Op Name', 'Task Duration(us)'),
               ('task_time*.csv', 'kernel_name', 'task_time(us)')]
    for pattern, name_key, duration_key in schemas:
        matches = sorted(root.rglob(pattern))
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f'ambiguous capture: {len(matches)} {pattern} reports')
        report = matches[0]
        groups = defaultdict(list)
        with report.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {name_key, duration_key}.issubset(reader.fieldnames or []):
                raise ValueError(f'unrecognized timing schema: {report}')
            for row in reader:
                if not row[name_key]:
                    raise ValueError('timing row missing operator identity')
                groups[row[name_key]].append(float(row[duration_key]))
        operators = [aggregate_rows(name, values, repeats) for name, values in groups.items()]
        if not operators or not any(item['mean_us'] > 0 for item in operators):
            raise ValueError('capture has no positive device timing')
        return {'latency_us': sum(item['mean_us'] for item in operators),
                'operators': operators, 'raw_csv': file_identity(report)}
    raise ValueError('no supported device duration CSV found')


def capture(fn, root, warmup, repeats, metric='PipeUtilization'):
    """NPU schedule and instrumentation match retained CANNBot capture code."""
    import torch
    import torch_npu

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        aic_metrics=getattr(profiler.AiCMetrics, metric),
        profiler_level=profiler.ProfilerLevel.Level1,
        l2_cache=False, data_simplification=False)
    # The output stays on the worker. No raw report/timeline transfer is implied.
    with profiler.profile(
        activities=[profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=warmup, active=repeats, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(root)),
        record_shapes=False, profile_memory=False, with_stack=False,
        with_flops=False, with_modules=False, experimental_config=experimental,
    ) as prof:
        for _ in range(warmup + repeats):
            fn()
            torch.npu.synchronize()
            prof.step()
    return root


def benchmark(fn, root, warmup=5, repeats=50, *, identity=None):
    """Benchmark a verified callable; inputs, device and fresh state are caller-owned."""
    if type(warmup) is not int or type(repeats) is not int or warmup < 0 or repeats < 1:
        raise ValueError('warmup >= 0 and repeats >= 1 must be integers')
    root = Path(root).resolve()
    raw = capture(fn, root / 'raw', warmup, repeats)
    result = parse_capture(raw, repeats)
    import torch
    import torch_npu
    result.update(schema_version=1, identity=identity or {}, warmup=warmup,
                  repeats=repeats, boundary='mean summed device-operator duration per invocation',
                  unit='us', instrumented=True, aggregation_policy=POLICY,
                  cannbot_source_sha256=CANNBOT_SHA256,
                  versions={'torch': torch.__version__, 'torch_npu': torch_npu.__version__},
                  device=str(torch.npu.current_device()), raw_report=str(raw))
    (root / 'timing.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
