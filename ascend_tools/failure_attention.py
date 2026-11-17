"""Read-only campaign attention overlay; immutable knowledge stays available."""
import json
from pathlib import Path
import sqlite3


def attention_snapshot(database, revision=None):
    ledger = (Path(database).parent / 'state.sqlite').resolve()
    if not ledger.is_file():
        return {'revision': 0, 'attempts': {}}
    with sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='failure_attention_actions'").fetchone():
            return {'revision': 0, 'attempts': {}}
        maximum = db.execute('SELECT COALESCE(MAX(sequence),0) FROM failure_attention_actions').fetchone()[0]
        chosen = maximum if revision is None else revision
        if type(chosen) is not int or not 0 <= chosen <= maximum:
            raise ValueError('Unknown attention revision')
        latest = {}
        for kind, identity, state, incident in db.execute('''SELECT e.kind,e.identity,e.attention,a.incident
                FROM failure_attention_effects e JOIN failure_attention_actions a ON a.sequence=e.action_sequence
                WHERE e.action_sequence<=? ORDER BY e.sequence''', (chosen,)):
            latest[kind, identity] = {'state': state, 'incident': incident}
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        attempts = {}
        for identity, project in db.execute('SELECT id,project FROM attempts'):
            state = latest.get(('attempt', identity), latest.get(('project', project)))
            if not state or state['state'] == 'active':
                continue
            if 'skill_backlog_occurrences' in tables:
                refs = [r[0] for r in db.execute('SELECT reference FROM skill_backlog_occurrences WHERE attempt=?', (identity,))]
                if any(latest.get(('occurrence', ref), {'state': 'active'})['state'] == 'active' for ref in refs):
                    continue
            # A useful kernel observation survives even a later harness failure.
            if db.execute("SELECT 1 FROM events WHERE attempt=? AND type='verification' LIMIT 1", (identity,)).fetchone():
                continue
            attempts[identity] = state
        return {'revision': chosen, 'attempts': attempts}


def lesson_attention(database, collection, snapshot):
    if not collection.startswith('lesson-') or not snapshot['attempts']:
        return None
    source = Path(str(database) + '.lessons') / (collection[7:] + '.json')
    try:
        record = json.loads(source.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # Structured, exact provenance only; never infer ownership from lesson prose.
    if record.get('delivery_id') != collection[7:] or record.get('outcome') != 'negative':
        return None
    return snapshot['attempts'].get(record.get('attempt_id'))
