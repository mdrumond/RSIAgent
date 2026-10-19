"""Discover runtime metric capabilities and curate compact non-timeline evidence."""
import csv
import json
import math
import re
import subprocess
from pathlib import Path

from .measurement import capture, file_identity

GROUPS = ('PipeUtilization', 'ArithmeticUtilization', 'Memory', 'MemoryL0',
          'MemoryUB', 'L2Cache', 'ResourceConflictRatio', 'MemoryAccess')

# CANN 9.1 msprof summary names observed on A3, plus retained long-name
# exports. Generic task times/cycles occur in every group and are not counters
# proving that the requested group was exported. Unknown schemas stay visible
# in the report but cannot establish counter availability.
COUNTER_SCHEMA = 'cann-9.1-summary-and-long-name-aliases-v1'
COUNTER_NAMES = {
    'PipeUtilization': r'(?:aic|aiv)_(?:mac|cube|vec|scalar|mte[123]|fixpipe)_ratio',
    'ArithmeticUtilization': r'(?:aic_mac|aiv_vec)_(?:fp\d+|bf16|int\d+|misc)_ratio|'
                             r'aic_cube_fops|aiv_vector_fops',
    'Memory': r'(?:aic|aiv)_(?:l1|l2|main_mem|ub)_(?:read|write)_bw|'
              r'(?:aic|aiv)_mte[123]_(?:main_mem|l2)_(?:read|write)_bandwidth|'
              r'(?:aic|aiv)_mte[123]_(?:read|write)_main_memory',
    'MemoryL0': r'(?:aic|aiv)_l0[abc]_(?:read|write)_bw(?:_cube)?|'
                r'(?:aic|aiv)_mte[123]_l0[abc]_bandwidth',
    'MemoryUB': r'(?:aic|aiv)_ub_(?:read|write)_bw_(?:scalar|vector)|'
                r'(?:aic|aiv)_(?:mte[123]|vec)_ub_bandwidth',
    'L2Cache': r'(?:aic|aiv)_(?:(?:r[01]_)?read|write)_cache_(?:hit|miss_allocate)|'
               r'(?:aic|aiv)_l2_?cache_(?:read|write)_hit_rate',
    'ResourceConflictRatio': r'(?:aic|aiv)_vec_(?:bankgroup|bank|resc)_cflt_ratio',
    'MemoryAccess': r'(?:aic|aiv)_(?:read|write)_main_memory_datas|'
                    r'(?:GM_to_L1|L0C_to_L1|L0C_to_GM|GM_to_UB|UB_to_GM)_datas|'
                    r'(?:aic|aiv)_mte[123]_(?:read|write)_main_memory',
}


def group_counter_columns(group, reports):
    """Return finite requested-group columns, including genuinely measured zero."""
    pattern = COUNTER_NAMES[group]
    return sorted({metric['column'] for report in reports for metric in report['metrics']
                   if metric['available'] and re.fullmatch(pattern, metric['name'], re.IGNORECASE)})


def discover(*, metrics=None, help_text=None):
    """Probe the installed CLI and Torch-NPU enum, retaining both capability sources."""
    if help_text is None:
        try:
            proc = subprocess.run(['msprof', '--help'], capture_output=True, text=True, timeout=20)
            help_text = proc.stdout + proc.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            help_text = f'unavailable: {exc}'
    if metrics is None:
        import torch_npu
        metrics = torch_npu.profiler.AiCMetrics
    supported = [group for group in GROUPS if hasattr(metrics, group)]
    return {'groups': sorted(supported), 'help': help_text,
            'capability_source': 'installed torch_npu AiCMetrics; capture confirms availability'}


def normalize_csv(path):
    """Keep vendor metric names/units and row identity; never invent ratio units."""
    result = []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            identity = {k: v for k, v in row.items()
                        if k and re.search(r'(^|[ _])(name|type|id)($|[ _])', k.lower())}
            for column, raw in row.items():
                if not column or column in identity:
                    continue
                text = (raw or '').strip()
                missing = text.lower() in ('', 'na', 'n/a', 'nan', 'inf', '-inf', '--')
                try:
                    value = float(text.rstrip('%')) if not missing else None
                except ValueError:
                    continue
                if value is not None and not math.isfinite(value):
                    value = None
                match = re.fullmatch(r'(.+?)\s*\(([^()]*)\)', column)
                name, unit = match.groups() if match else (column, '%' if text.endswith('%') else None)
                result.append({'row': index, 'identity': identity, 'column': column,
                               'name': name.strip(), 'unit': unit, 'value': value,
                               'available': value is not None, 'raw': raw})
    return result


def compact_reports(raw_root, output):
    """Only summary CSVs are eligible. Timeline/task-trace exports stay remote."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(Path(raw_root).rglob('*.csv')):
        if not path.name.startswith(('op_summary', 'kernel_details')):
            continue
        rows = normalize_csv(path)
        target = output / f'metrics-{len(reports)}.json'
        target.write_text(json.dumps(rows, indent=2, allow_nan=False))
        csv_target = target.with_suffix('.csv')
        with csv_target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['row', 'identity', 'column', 'name',
                                                       'unit', 'value', 'available', 'raw'])
            writer.writeheader()
            writer.writerows({**r, 'identity': json.dumps(r['identity'], sort_keys=True)} for r in rows)
        reports.append({'source': file_identity(path), 'json': file_identity(target),
                        'csv': file_identity(csv_target), 'metrics': rows})
    return reports


def profile(fn, root, groups=None, *, identity=None):
    """Capture separate diagnostic replays. Failures never replace canonical timing."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    capabilities = discover()
    results = []
    for group in (GROUPS if groups is None else groups):
        item = {'group': group, 'available': False, 'reports': [],
                'counter_schema': COUNTER_SCHEMA, 'counter_columns': []}
        if group not in GROUPS or group not in capabilities['groups']:
            item['reason'] = 'unsupported non-timeline metric group'
        else:
            try:
                raw = capture(fn, root / group / 'raw', 5, 1, metric=group)
                item.update(raw_report=str(raw), reports=compact_reports(raw, root / group / 'compact'))
                item['counter_columns'] = group_counter_columns(group, item['reports'])
                item['available'] = bool(item['counter_columns'])
                if not item['available']:
                    item['reason'] = ('no group counters exported' if item['reports']
                                      else 'capture exported no supported summary CSV')
            except Exception as exc:
                item['reason'] = f'{type(exc).__name__}: {exc}'
        results.append(item)
    result = {'schema_version': 1, 'identity': identity or {}, 'diagnostic': True,
              'capabilities': capabilities, 'groups': results, 'raw_report_root': str(root),
              'transfer_policy': 'compact CSV/JSON only; raw reports retained remotely'}
    (root / 'profile.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
