"""Read-only projections of retained learning experiments and active clocks."""
import hashlib
import json
from pathlib import Path

from flask import abort, jsonify


def read(path, default=None):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return default


def lesson_id(lesson):
    return hashlib.sha256(json.dumps(lesson, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def experiments(root):
    rows = []
    for directory in sorted((Path(root) / 'memory-usefulness').glob('*')):
        if not directory.is_dir():
            continue
        events = [event for path in sorted(directory.glob('[0-9]*.json'))
                  if isinstance(event := read(path), dict)]
        if not events:
            continue
        start = events[0]
        assessment = next((event for event in reversed(events) if event.get('kind') == 'assessment'), {})
        frozen = next((event for event in events if event.get('kind') == 'project-frozen'), {})
        arms = dict(assessment.get('arms', {}))
        for event in events:
            if event.get('kind') == 'arm-completed':
                arms[event['arm']] = event
        rows.append(dict(experiment_id=directory.name, lesson_id=start.get('lesson_id'),
            source_id=start.get('source_id'), lesson=start.get('lesson', {}),
            outcome=assessment.get('outcome', 'pending'), arms=arms,
            reason=assessment.get('reason'), interpretation=assessment.get('interpretation'),
            project=frozen.get('project'), arm_seconds=start.get('arm_seconds'),
            pair_seconds=start.get('pair_seconds')))
    return rows


def enrich(campaign):
    root = Path(campaign['root'])
    campaign['usefulness'] = experiments(root)
    campaign['time_budget'] = read(root / 'active-time.json')
    campaign['remediation'] = read(root / 'remediation.json', {})
    kinds = {}
    for row in read(root / 'failure-investigations.json', []):
        for key, role in (('project_id', 'remedial probe'), ('control_project_id', 'remedial control')):
            if row.get(key):
                kinds[row[key]] = dict(kind=role, feedback_id=row.get('feedback_id'))
    for project in campaign['projects']:
        source = kinds.get(project.get('project_id', project['id']), {})
        project['kind'] = source.get('kind', project.get('kind', 'new'))
        project['feedback_id'] = source.get('feedback_id')
        for lesson in project['memory']['lessons']:
            identity = lesson_id(lesson)
            lesson['usefulness'] = [row for row in campaign['usefulness'] if row['lesson_id'] == identity]
    return campaign


def register_usefulness(app, controls):
    def root_for(phase):
        entry = next((row for row in controls.campaigns.values() if row.get('phase') == phase), None)
        return Path(entry['root']) if entry else abort(404)

    @app.get('/api/phase/<int:phase>/usefulness')
    def usefulness_records(phase):
        return jsonify(experiments(root_for(phase)))

    @app.get('/api/phase/<int:phase>/usefulness/<identifier>/<arm>')
    def usefulness_evidence(phase, identifier, arm):
        if len(identifier) != 64 or any(ch not in '0123456789abcdef' for ch in identifier) or arm not in {'control', 'treatment'}:
            abort(404)
        path = root_for(phase) / 'memory-usefulness' / identifier / arm / 'evidence.json'
        if not path.is_file():
            abort(404)
        data = path.read_bytes()
        return jsonify(text=data[:8192].decode('utf-8', errors='replace'), bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest(), truncated=len(data) > 8192)
