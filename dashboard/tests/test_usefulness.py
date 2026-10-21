import json

from dashboard.usefulness import enrich, experiments, lesson_id
from benchmarks.ascend.training_annex import _retained


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_readonly_missing_roots(tmp_path):
    root = tmp_path / 'missing'
    assert experiments(root) == []
    campaign = enrich(dict(root=str(root), projects=[]))
    assert campaign['time_budget'] is None
    assert not root.exists()


def test_lesson_pair_lineage_and_portable_journal(tmp_path):
    lesson = dict(category='shape', claim='Handle tails')
    directory = tmp_path / 'memory-usefulness' / ('a' * 64)
    write(directory / '0000.json', dict(kind='started', lesson=lesson,
        lesson_id=lesson_id(lesson), source_id='original', pair_seconds=600, arm_seconds=300))
    write(directory / '0001.json', dict(kind='arm-completed', arm='control', verified=True))
    write(directory / '0002.json', dict(kind='assessment', outcome='both-succeeded',
        arms=dict(control=dict(verified=True), treatment=dict(verified=True)), interpretation='Uncertain'))
    write(directory / 'control/evidence.json', dict(verified=True))
    write(tmp_path / 'failure-investigations.json', [dict(project_id='probe',
        control_project_id='control', feedback_id='failure')])
    result = enrich(dict(root=str(tmp_path), projects=[dict(id='probe', memory=dict(lessons=[lesson]))]))
    project = result['projects'][0]
    assert project['kind'] == 'remedial probe'
    assert project['feedback_id'] == 'failure'
    assert project['memory']['lessons'][0]['usefulness'][0]['outcome'] == 'both-succeeded'
    retained = dict(_retained(tmp_path))
    assert str((directory / '0000.json').relative_to(tmp_path)) in retained
    assert str((directory / 'control/evidence.json').relative_to(tmp_path)) in retained
