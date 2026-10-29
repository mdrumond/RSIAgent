import fcntl
import hashlib
import json

import pytest

from dashboard.app import create_app
from dashboard.subjects import SubjectViews
from benchmarks.a5kernels.sft_dataset import TraceRecorder, _read_trace


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'subject'
    plan = dict(project_id='copy', spec=dict(hypothesis='Inspect the copy tail.'))
    write(root / 'subject-campaign.json', dict(waves=['wave-1'], projects={
        'copy': dict(plan=plan, wave_index=1, objective='Copy',
                     outcome=dict(host_verified=True, artifacts={}),
                     review=dict(decision='accept', memory=dict(summary='Tail handling')))},
        pending=dict(wave_id='wave-2', subject=dict(subject_id='tails'),
                     projects=['live', 'queued'], items=[dict(plan, project_id='live'),
                                                        dict(plan, project_id='queued')])))
    write(root / 'subjects.json', dict(subjects=[dict(subject_id='tails', objective='Tail handling')],
        waves={'wave-1': dict(subject_id='tails'), 'wave-2': dict(subject_id='tails')}))
    write(root / 'memory/active.json', dict(admissions={'copy': dict(version='a' * 64)}))
    live = root / 'waves/wave-2/live'
    live.mkdir(parents=True)
    config = tmp_path / 'config.json'
    write(config, dict(subject_campaigns=[dict(id='research', root=str(root), target='a3')]))
    return config, root, live


def test_subject_projection_distinguishes_pending_stopped_verified_and_publication(campaign):
    config, root, _ = campaign
    views = SubjectViews(config)
    rows = {p['id']: p for p in views.campaign('research')['projects']}
    assert rows['queued']['execution'] == 'pending'
    assert rows['live']['execution'] == 'stopped'
    assert rows['copy']['execution'] == 'verified'
    assert rows['copy']['project_review'] == 'accept'
    assert rows['copy']['merged'] is True
    assert rows['copy']['wave_id'] == 'wave-1'
    with (root / '.campaign.lock').open('wb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert views.project('research', 'live')[1]['execution'] == 'running'


def test_incremental_logs_wait_for_complete_records_and_reconnect(campaign):
    config, _, live = campaign
    path = live / 'evidence.jsonl'
    first = dict(kind='message', payload=dict(content='first'))
    line = json.dumps(first).encode() + b'\n'
    path.write_bytes(line + b'{"kind":')
    views = SubjectViews(config)
    page = views.logs('research', 'live')
    assert [e['event'] for e in page['events']] == [first]
    assert views.logs('research', 'live', page['cursor'])['events'] == []
    with path.open('ab') as stream:
        stream.write(b'"tool","payload":{"status":"passed"}}\n')
    next_page = views.logs('research', 'live', page['cursor'])
    assert [e['event']['kind'] for e in next_page['events']] == ['tool']
    assert next_page['events'][0]['offset'] == len(line)
    assert views.logs('research', 'live', next_page['cursor'])['events'] == []
    path.write_text(json.dumps(dict(kind='replacement', payload={})) + '\n')
    reset = views.logs('research', 'live', next_page['cursor'])
    assert reset['reset'] is True
    assert reset['events'][0]['event']['kind'] == 'replacement'


def test_log_routes_expand_exact_payloads_redact_and_download(campaign, tmp_path):
    config, _, live = campaign
    payload = json.dumps(dict(content='model answer', password='hidden')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = live / 'artifacts/objects' / digest[:2] / digest[2:]
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    (live / 'evidence.jsonl').write_text(json.dumps(dict(kind='artifact', payload=dict(sha256=digest))) + '\n')
    app = create_app(config, tmp_path / 'cache.sqlite')
    client = app.test_client()
    try:
        assert client.get('/').status_code == 200
        assert client.get('/campaign/research').status_code == 200
        assert client.get('/campaign/research/project/live').status_code == 200
        base = '/api/campaigns/research/projects/live/logs'
        event = client.get(base).get_json()['events'][0]['event']
        assert event['retained_payload'] == dict(content='model answer', password='[REDACTED]')
        downloaded = client.get(base + '/download')
        assert json.loads(downloaded.data)['event'] == event
        assert client.get(base + '?cursor=invalid').status_code == 400
        assert client.get(base + '?limit=0').status_code == 400
        assert client.get('/api/campaigns/missing').status_code == 404
    finally:
        app.extensions['dashboard_controls'].close()


def test_large_logs_tail_then_load_earlier_without_duplicates(campaign):
    config, _, live = campaign
    rows = [json.dumps(dict(kind='message', payload=dict(content=str(i) * 1000))) + '\n'
            for i in range(200)]
    (live / 'evidence.jsonl').write_text(''.join(rows))
    views = SubjectViews(config)
    tail = views.logs('research', 'live', limit=500)
    assert tail['earlier'] is True
    assert 0 < len(tail['events']) < 200
    history = views.logs('research', 'live', 'start', 500)
    assert len(history['events']) == 200
    assert {e['id'] for e in tail['events']} <= {e['id'] for e in history['events']}
    assert views.logs('research', 'live', tail['cursor'])['events'] == []


def test_trace_messages_are_visible_before_finish_and_final_seal_remains_valid(tmp_path):
    path = tmp_path / 'traces/actor.jsonl'
    recorder = TraceRecorder(path, trace_id='actor', phase='phase1', round_id='round',
        operator_family='copy', model='model', source_manifest_sha256='a' * 64)
    recorder.message('assistant', 'inspect the tail')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]['payload']['content'] == 'inspect the tail'
    recorder.validation(candidate_id='candidate', validator='host', status='passed',
                        evidence_sha256='b' * 64, candidate_sha256='c' * 64)
    recorder.finish()
    assert _read_trace(path)[-1]['kind'] == 'terminal'


def test_log_reset_identifies_only_the_replaced_stream(campaign):
    config, _, live = campaign
    first = live / 'evidence.jsonl'
    other = live / 'sft-traces/other.jsonl'
    other.parent.mkdir()
    first.write_text(json.dumps(dict(kind='message', payload='old actor entry')) + '\n')
    other.write_text(json.dumps(dict(kind='tool', payload='retained host observation')) + '\n')
    views = SubjectViews(config)
    initial = views.logs('research', 'live', 'start')
    assert len(initial['events']) == 2
    first.write_text(json.dumps(dict(kind='message', payload='replacement')) + '\n')
    page = views.logs('research', 'live', initial['cursor'])
    stream = first.relative_to(live.parents[2]).as_posix()
    assert page['reset_streams'] == [stream]
    assert len(page['events']) == 1 and page['events'][0]['event']['payload'] == 'replacement'
    retained = {row['id']: row for row in initial['events'] if row['stream'] not in page['reset_streams']}
    retained.update((row['id'], row) for row in page['events'])
    assert {row['event']['payload'] for row in retained.values()} == {'replacement', 'retained host observation'}
    assert views.logs('research', 'live', page['cursor'])['events'] == []


def test_live_log_client_retains_other_streams_when_one_rotates():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required to execute the dashboard client behavior')
    script = """
const assert = require('assert');
const vm = require('vm');
const fs = require('fs');
const context = {document: {querySelectorAll: () => []}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const entries = new Map();
context.updateActorLogEntries(entries, {events: [
  {id: 'actor:0', stream: 'actor', event: {content: 'old'}},
  {id: 'host:0', stream: 'host', event: {content: 'retained'}}], reset_streams: []});
context.updateActorLogEntries(entries, {events: [
  {id: 'actor:0', stream: 'actor', event: {content: 'replacement'}}], reset_streams: ['actor']});
assert.strictEqual(entries.size, 2);
assert.strictEqual(entries.get('host:0').event.content, 'retained');
assert.strictEqual(entries.get('actor:0').event.content, 'replacement');
"""
    subprocess.run([node, '-e', script, str(Path(__file__).parents[1] / 'static/subject_logs.js')], check=True)


def test_oversized_complete_record_advances_to_following_events(campaign):
    config, _, live = campaign
    oversized = json.dumps(dict(kind='message', payload='x' * (1048576 + 100))).encode() + b'\n'
    following = dict(kind='tool', payload='next action')
    (live / 'evidence.jsonl').write_bytes(oversized + json.dumps(following).encode() + b'\n')
    views = SubjectViews(config)
    first = views.logs('research', 'live', 'start', limit=1)
    assert first['events'][0]['event']['kind'] == 'oversized'
    assert first['events'][0]['event']['payload']['bytes'] == len(oversized)
    next_page = views.logs('research', 'live', first['cursor'])
    assert [row['event'] for row in next_page['events']] == [following]
    assert views.logs('research', 'live', next_page['cursor'])['events'] == []


def test_large_partial_tail_is_retained_until_newline_then_following_rows_flow(campaign):
    config, _, live = campaign
    path = live / 'evidence.jsonl'
    prefix = json.dumps(dict(kind='message', payload='older event')).encode() + b'\n'
    partial = json.dumps(dict(kind='message', payload='x' * (1048576 + 100))).encode()
    path.write_bytes(prefix + partial)
    views = SubjectViews(config)
    # A fresh tail poll lands inside the long unfinished row. It must retain
    # that row's start rather than bookmarking its temporary EOF.
    initial = views.logs('research', 'live')
    assert initial['events'] == []
    assert views.logs('research', 'live', initial['cursor'])['events'] == []
    following = dict(kind='tool', payload='after completed long record')
    with path.open('ab') as stream:
        stream.write(b'\n' + json.dumps(following).encode() + b'\n')
    complete = views.logs('research', 'live', initial['cursor'])
    assert [row['event']['kind'] for row in complete['events']] == ['oversized', 'tool']
    assert complete['events'][0]['event']['payload']['bytes'] == len(partial) + 1
    assert complete['events'][1]['event'] == following
    assert complete['reset_streams'] == []
