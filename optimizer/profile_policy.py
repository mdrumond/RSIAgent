"""Bound diagnostic replays independently of the full correctness/timing suite."""
from .profiles import GROUPS


def policy(value=None):
    value = {} if value is None else value
    count = value.get('max_cases', 1)
    groups = value.get('groups', ['PipeUtilization'])
    if type(count) is not int or not 1 <= count <= 24:
        raise ValueError('profiling max_cases must be between one and 24')
    if not isinstance(groups, (list, tuple)) or not groups or any(g not in GROUPS for g in groups):
        raise ValueError('profiling groups must be known non-timeline metric groups')
    return {'max_cases': count, 'groups': list(dict.fromkeys(groups))}


def select(prepared, value=None):
    """Largest frozen input volume first; ties retain frozen-suite order."""
    configured = policy(value)
    def volume(row):
        return sum(x.numel() for x in [*row['args'], *row['kwargs'].values()]
                   if hasattr(x, 'numel'))
    ranked = sorted(prepared.cases, key=volume, reverse=True)
    selected = ranked[:configured['max_cases']]
    return selected, {'policy': configured, 'selection': 'largest-input-numel-first; stable ties',
                      'suite_sha256': prepared.manifest['suite_sha256'],
                      'suite_case_count': len(prepared.cases),
                      'selected_cases': [row['case']['id'] for row in selected],
                      'diagnostic_only': True,
                      'correctness_scope': 'selected cases only; use separate full-suite evaluation'}
