"""Lazy projections of immutable optimizer evidence; never open remote report trees."""
import difflib
import json
import math

from flask import abort, jsonify, Response


def positive(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def sources(store, key):
    return [row for row in store.artifacts(key) if row['name'] == 'candidate.cpp']


def projection(store, key):
    store.get(key)
    events = store.events(key)
    state = next((e['payload'] for e in reversed(events)
                  if e['kind'] == 'optimizer-state'), {})
    def read(record):
        return json.loads(store.read_artifact(key, record['sha256'])) if record else {}
    baseline, best = read(state.get('baseline')), state.get('best')
    history = [row for row in state.get('history', []) if 'candidate_sha256' in row
               and row.get('status') != 'duplicate-skipped']
    attempts, maximum = [], None
    for index, row in enumerate(history, 1):
        value = positive(row.get('speedup'))
        maximum = max(maximum or 0, value) if value is not None else maximum
        attempts.append({'attempt': index, 'speedup': value, 'best_speedup': maximum,
                         'summary': row.get('summary'), 'candidate_sha256': row['candidate_sha256'],
                         'status': row.get('content', {}).get('status')})
    actual = {row['case']: row for row in (best or {}).get('content', {}).get('cases', [])}
    cases = []
    for row in baseline.get('cases', []):
        a = positive(row.get('timing', {}).get('torch', {}).get('latency_us'))
        b = positive(actual.get(row['case'], {}).get('timing', {}).get('ascendc', {}).get('latency_us'))
        cases.append({'case': row['case'], 'torch_us': a, 'ascendc_us': b,
                      'speedup': a / b if a and b else None})
    metrics, unavailable = [], []
    def visit(node, context):
        if isinstance(node, list):
            for child in node:
                visit(child, context)
        elif isinstance(node, dict):
            context = {**context, **({'case': node['case']} if 'case' in node else {})}
            if node.get('status') == 'unavailable':
                unavailable.append({**context, 'group': 'profiling',
                                    'reason': node.get('error', node.get('reason', 'unavailable'))})
            if 'groups' in node:
                for group in node['groups']:
                    if not group.get('available'):
                        unavailable.append({**context, 'group': group['group'],
                                            'reason': group.get('reason', 'unavailable')})
                    for report in group.get('reports', []):
                        for metric in report.get('metrics', []):
                            metrics.append({**metric, **context, 'group': group['group']})
                return
            for name, child in node.items():
                visit(child, {**context, **({'engine': name} if name in {'torch', 'ascendc'} else {})})
    visit(baseline.get('profiling', {}), {'engine': 'torch'})
    seen_profiles = set()
    for event in events:
        previous = event['payload'].get('best', {}) if event['kind'] == 'optimizer-state' else {}
        record = previous.get('profiling')
        if record and record['sha256'] not in seen_profiles:
            seen_profiles.add(record['sha256'])
            attempt = next((i for i, row in enumerate(history, 1)
                            if (row.get('configuration_sha256') == previous['configuration_sha256']
                                if previous.get('configuration_sha256') else
                                row['candidate_sha256'] == previous['candidate_sha256'])), None)
            visit(read(record), {'engine': 'ascendc', 'attempt': attempt,
                                 'candidate_sha256': previous['candidate_sha256']})
    return {'cases': cases, 'attempts': attempts, 'best': None if not best else {
                name: best.get(name) for name in ('speedup', 'candidate_sha256', 'summary', 'faster_than_torch')},
            'metrics': metrics, 'unavailable': unavailable, 'sources': [
                {'index': i, **row} for i, row in enumerate(sources(store, key))]}


def register_results(app, manager):
    @app.get('/api/optimizer/jobs/<job_id>/artifacts/<digest>')
    def optimizer_artifact(job_id, digest):
        if not manager.enabled:
            abort(403)
        try:
            registered = {row['sha256']: row for row in manager.store.artifacts(job_id)}
            if digest not in registered:
                abort(404)
            data = manager.store.read_artifact(job_id, digest)
            return Response(data, mimetype='text/plain')
        except (ValueError, OSError):
            abort(404)

    @app.get('/api/optimizer/jobs/<job_id>/results')
    def optimizer_results(job_id):
        if not manager.enabled:
            abort(403)
        try:
            return jsonify(projection(manager.store, job_id))
        except (ValueError, OSError, KeyError):
            abort(404)

    @app.get('/api/optimizer/jobs/<job_id>/sources/<int:index>')
    @app.get('/api/optimizer/jobs/<job_id>/sources/<int:index>/download')
    def optimizer_source(job_id, index):
        from flask import request
        if not manager.enabled:
            abort(403)
        try:
            rows = sources(manager.store, job_id)
            def load(i):
                source = json.loads(manager.store.read_artifact(job_id, rows[i]['sha256']))
                if not isinstance(source, str):
                    raise ValueError('invalid source artifact')
                return source
            source = load(index)
            if request.path.endswith('/download'):
                return Response(source, mimetype='text/plain', headers={
                    'Content-Disposition': f'attachment; filename="candidate-{index + 1}.cpp"'})
            previous = load(index - 1) if index else ''
            diff = ''.join(difflib.unified_diff(previous.splitlines(True), source.splitlines(True),
                                               fromfile='previous.cpp', tofile='candidate.cpp'))
            from pygments import highlight
            from pygments.formatters import HtmlFormatter
            from pygments.lexers import CppLexer
            return jsonify(source=source, diff=diff,
                           html=highlight(source, CppLexer(), HtmlFormatter()))
        except (ValueError, OSError, IndexError):
            abort(404)
