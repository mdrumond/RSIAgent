"""Published prefixes retain all instrumentation with linear reference I/O."""
import json
from pathlib import Path

import pytest

from core.artifacts import canonical_bytes
from rsi import event_journal as module
from rsi.event_journal import EventJournal, read_event_refs
from rsi.python_execution import atomic_json


def refs(count):
    return [f'{i:064x}' for i in range(count)]


def test_legacy_prefix_migration_reopen_checkpoint_and_duplicate_events(tmp_path):
    initial = refs(3)
    atomic_json(tmp_path / 'events.json', initial)
    journal = EventJournal(tmp_path)
    journal.append(initial[-1])  # Identical payloads are distinct observed events.
    assert read_event_refs(tmp_path) == initial + [initial[-1]]
    reopened = EventJournal(tmp_path)
    reopened.append(refs(5)[-1])
    reopened.checkpoint()
    expected = initial + [initial[-1], refs(5)[-1]]
    assert json.loads((tmp_path / 'events.json').read_bytes()) == expected
    assert EventJournal(tmp_path).refs == expected


@pytest.mark.parametrize('failure', ['partial_write', 'before_publication'])
def test_interrupted_append_never_publishes_tail_and_resume_does_not_duplicate(tmp_path, monkeypatch, failure):
    journal = EventJournal(tmp_path)
    a, b, c = refs(3)
    journal.append(a)
    real_write, real_atomic = module.os.write, module.atomic_json
    if failure == 'partial_write':
        calls = []
        def interrupted(fd, data):
            if calls:
                raise OSError('interrupted record')
            calls.append(True)
            return real_write(fd, data[:9])
        monkeypatch.setattr(module.os, 'write', interrupted)
    else:
        def interrupted(path, value):
            if Path(path).name == 'events.head.json':
                raise OSError('unpublished head')
            return real_atomic(path, value)
        monkeypatch.setattr(module, 'atomic_json', interrupted)
    with pytest.raises(OSError):
        journal.append(b)
    assert read_event_refs(tmp_path) == [a]
    with pytest.raises(RuntimeError):
        journal.append(c)
    monkeypatch.setattr(module.os, 'write', real_write)
    monkeypatch.setattr(module, 'atomic_json', real_atomic)
    reopened = EventJournal(tmp_path)
    reopened.append(c)
    assert read_event_refs(tmp_path) == [a, c]
    assert (tmp_path / 'events.refs').read_bytes() == (a + '\n' + c + '\n').encode()


def test_truncated_committed_prefix_fails_closed(tmp_path):
    journal = EventJournal(tmp_path)
    journal.append(refs(1)[0])
    (tmp_path / 'events.refs').write_bytes(b'123')
    with pytest.raises(ValueError):
        read_event_refs(tmp_path)


def test_hot_writer_does_not_invalidate_older_committed_snapshot(tmp_path, monkeypatch):
    journal = EventJournal(tmp_path)
    journal.append(refs(1)[0])
    real_head = module._head
    inside = False
    def advancing(root):
        nonlocal inside
        captured = real_head(root)
        if not inside:
            inside = True
            try:
                journal.append(f'{len(journal.refs):064x}')
                journal.checkpoint()  # Checkpoint can also advance past captured head.
            finally:
                inside = False
        return captured
    monkeypatch.setattr(module, '_head', advancing)
    for _ in range(20):
        expected = list(journal.refs)
        assert read_event_refs(tmp_path) == expected
        assert len(journal.refs) == len(expected) + 1


def test_reference_write_bytes_grow_linearly_not_per_delta_full_index(tmp_path, monkeypatch):
    written = dict(journal=0, head=0, checkpoint=0)
    real_write, real_atomic = module.os.write, module.atomic_json
    def count_write(fd, data):
        result = real_write(fd, data)
        written['journal'] += result
        return result
    def count_atomic(path, value):
        category = 'head' if Path(path).name == 'events.head.json' else 'checkpoint'
        result = real_atomic(path, value)
        written[category] += Path(path).stat().st_size
        return result
    monkeypatch.setattr(module.os, 'write', count_write)
    monkeypatch.setattr(module, 'atomic_json', count_atomic)
    journal = EventJournal(tmp_path)
    count = 2048
    for reference in refs(count):
        journal.append(reference)
    journal.checkpoint()
    assert read_event_refs(tmp_path) == refs(count)
    assert written['journal'] == count * 65
    assert sum(written.values()) < count * 400
    quadratic = 68 * count * (count + 1) // 2
    assert quadratic > 100 * sum(written.values())
    (tmp_path / 'write-evidence.json').write_text(json.dumps(dict(events=count,
        serialized_bytes=written, total=sum(written.values()),
        previous_full_index_bytes=quadratic), indent=2))


def test_project_runner_reopen_retains_stream_usage_and_typed_host_events(tmp_path):
    from test_rsi_project_runner import make
    from rsi.project_runner import ProjectRunner
    runner, injections, database = make(tmp_path)
    events = [dict(type='pi-event', event=dict(type='message_update', delta='complete raw delta')),
              dict(type='skill-usage', skill='debug', action='read'),
              dict(type='host-dispatch', operation_id='owned', receipt={'kind': 'typed'}),
              dict(type='host-result', operation_id='owned', receipt={'status': 'failed'})]
    try:
        for event in events:
            runner.append(event)
        reopened = ProjectRunner(runner.root, **injections)
        assert reopened.events == runner.events == events
        assert reopened.event_refs == runner.event_refs
        assert not reopened.no_host_effects('owned')
    finally:
        database.close()


def test_export_includes_uncheckpointed_stream_and_usage_in_order(tmp_path):
    from test_rsi_evidence_io import task_fixture
    from rsi.evidence_io import export_task
    task, store, *_ = task_fixture(tmp_path)
    before = read_event_refs(task)
    journal = EventJournal(task)
    events = [dict(type='pi-event', event=dict(type='delta', text='all raw text')),
              dict(type='skill-usage', action='read', skill='debug')]
    for event in events:
        journal.append(store.put_json(event))
    assert json.loads((task / 'events.json').read_bytes()) == before
    destination = tmp_path / 'export'
    export_task(task, destination)
    rows = [json.loads(line) for line in (destination / 'evidence.jsonl').read_text().splitlines()]
    exported = [row['data'] for row in rows if row['kind'] == 'event']
    assert exported[-2:] == events
    assert len(exported) == len(before) + len(events)
    publication = next(row['data'] for row in rows if row['kind'] == 'state' and row['id'] == 'event-journal')
    assert publication['head']['count'] == len(exported)
    assert publication['committed_refs'].splitlines() == journal.refs[len(before):]
