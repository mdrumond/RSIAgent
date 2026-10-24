import json

import pytest

from benchmarks.ascend.subjects import Subject, SubjectCurriculumAgent, SubjectScheduler, load_subjects


SNAPSHOT = 'a' * 64


def decide(scheduler, action, **fields):
    return scheduler.decide(dict(action=action, reason='Observed coverage', evidence=['evidence:1'], **fields))


def wave(scheduler, wave_id, reviews=()):
    scheduler.start_wave(wave_id, SNAPSHOT)
    scheduler.complete_wave(wave_id, reviews)


def test_minimum_and_configuration_resume(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'subjects': [{'subject_id': 'copy', 'objective': 'Correct copies'}]}))
    subjects = load_subjects(path)
    scheduler = SubjectScheduler(tmp_path / 'run', subjects)
    wave(scheduler, 'one')
    with pytest.raises(ValueError, match='minimum'):
        decide(scheduler, 'advance')
    assert scheduler.context()['completed_waves'] == {'copy': 1}
    resumed = SubjectScheduler(tmp_path / 'run', subjects)
    wave(resumed, 'two')
    wave(resumed, 'three')
    decide(resumed, 'advance')
    assert resumed.is_complete()
    with pytest.raises(ValueError, match='configuration changed'):
        SubjectScheduler(tmp_path / 'run', [Subject('copy', 'Different objective')])


def test_wave_snapshot_and_completion_are_idempotent(tmp_path):
    scheduler = SubjectScheduler(tmp_path, [Subject('copy', 'Correct copies', 1)])
    first = scheduler.start_wave('one', SNAPSHOT)
    assert scheduler.start_wave('one', SNAPSHOT) == first
    with pytest.raises(ValueError, match='snapshot changed'):
        scheduler.start_wave('one', 'b' * 64)
    with pytest.raises(ValueError, match='minimum'):
        decide(scheduler, 'advance')
    completed = scheduler.complete_wave('one', ['review'])
    assert scheduler.complete_wave('one', ['review']) == completed
    with pytest.raises(ValueError, match='reviews changed'):
        scheduler.complete_wave('one')
    assert scheduler.context()['completed_waves'] == {'copy': 1}


def test_provisional_progress_keeps_old_snapshot_and_waits_for_reviews(tmp_path):
    subjects = [Subject('copy', 'Copies', 1), Subject('reduce', 'Reductions', 1)]
    scheduler = SubjectScheduler(tmp_path, subjects)
    wave(scheduler, 'one', ['memory-one'])
    decide(scheduler, 'advance')
    record = scheduler.start_wave('two', SNAPSHOT)
    assert record['subject_id'] == 'reduce'
    scheduler.resolve_review('memory-one', 'accepted')
    assert scheduler.state()['waves']['two']['memory_snapshot_sha256'] == SNAPSHOT
    scheduler.complete_wave('two', ['memory-two'])
    decide(scheduler, 'advance')
    assert scheduler.context()['current_subject'] is None
    assert not scheduler.is_complete()
    scheduler.resolve_review('memory-two', 'rejected')
    assert scheduler.is_complete()
    with pytest.raises(ValueError, match='different outcome'):
        scheduler.resolve_review('memory-two', 'accepted')


def test_repairs_block_readiness_and_revisit_returns_to_frontier(tmp_path):
    subjects = [Subject('copy', 'Copies', 1), Subject('reduce', 'Reductions', 1)]
    scheduler = SubjectScheduler(tmp_path, subjects)
    wave(scheduler, 'one')
    scheduler.add_repair('repair-one', 'copy', 'project-one', 'Accidental success', ['run:1'])
    with pytest.raises(ValueError, match='open repairs'):
        decide(scheduler, 'advance')
    decide(scheduler, 'repair', repair_id='repair-one')
    wave(scheduler, 'repair-wave')
    scheduler.resolve_repair('repair-one', ['run:2'])
    decide(scheduler, 'advance')
    decide(scheduler, 'revisit', subject_id='copy')
    wave(scheduler, 'revisit-wave')
    decide(scheduler, 'advance')
    assert scheduler.context()['current_subject']['subject_id'] == 'reduce'
    wave(scheduler, 'last')
    decide(scheduler, 'advance')
    assert scheduler.is_complete()


def test_late_repair_reopens_completed_progress(tmp_path):
    scheduler = SubjectScheduler(tmp_path, [Subject('copy', 'Copies', 1)])
    wave(scheduler, 'one')
    decide(scheduler, 'advance')
    scheduler.add_repair('late', 'copy', 'project-one', 'Untested tail', ['run:1'])
    assert not scheduler.is_complete()
    resumed = SubjectScheduler(tmp_path, [Subject('copy', 'Copies', 1)])
    decide(resumed, 'repair', repair_id='late')
    wave(resumed, 'extra')
    resumed.resolve_repair('late', ['tail:pass'])
    decide(resumed, 'advance')
    assert resumed.is_complete()


def test_agent_dispatches_real_decision_and_propagates_context(tmp_path):
    scheduler = SubjectScheduler(tmp_path, [Subject('copy', 'Copies', 1)])
    wave(scheduler, 'one')
    requests = []

    def complete(messages, *, remaining_seconds):
        context = json.loads(messages[-1]['content'])
        requests.append((context['memory'], context['completed_waves'], remaining_seconds))
        return json.dumps(dict(action='advance', reason='Demonstrated copies', evidence=['run:1']))

    SubjectCurriculumAgent(complete).decide(scheduler, memory_context={'accepted': ['memory:1']}, remaining_seconds=20)
    assert scheduler.is_complete()
    assert requests == [({'accepted': ['memory:1']}, {'copy': 1}, 20)]


def test_invalid_decisions_preserve_state_and_do_not_skip_subjects(tmp_path):
    scheduler = SubjectScheduler(tmp_path, [Subject('copy', 'Copies'), Subject('reduce', 'Reduce')])
    before = scheduler.state()
    with pytest.raises(ValueError, match='previously reached'):
        decide(scheduler, 'revisit', subject_id='reduce')
    assert scheduler.state() == before
    with pytest.raises(ValueError, match='evidence'):
        scheduler.decide(dict(action='advance', reason='Ready', evidence=[]))
    assert scheduler.state() == before
