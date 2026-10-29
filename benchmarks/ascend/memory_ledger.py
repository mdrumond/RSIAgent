"""Shared, versioned memory access, judgments, and withdrawal history."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3

from .subject_memory import digest, StaleMemory


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _evidence(values):
    if not isinstance(values, list) or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in values):
        raise ValueError('memory evidence must contain artifact SHA256 identities')
    return sorted(set(values))


class MemoryLedger:
    """One durable companion database for every user of the same reference KDB.

    Keeping this ledger separate leaves immutable reference snapshots untouched.
    Each operation opens its own transaction, so processes and threads share the
    same access identities and lifecycle decisions without copying counters.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS memory_versions (
                    version TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                    record TEXT NOT NULL, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_heads (
                    memory_id TEXT PRIMARY KEY, version TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_lineage (
                    child TEXT NOT NULL, parent TEXT NOT NULL,
                    PRIMARY KEY(child,parent));
                CREATE TABLE IF NOT EXISTS memory_artifacts (
                    version TEXT NOT NULL, sha TEXT NOT NULL,
                    PRIMARY KEY(version,sha));
                CREATE TABLE IF NOT EXISTS memory_accesses (
                    event_id TEXT PRIMARY KEY, exposure_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL, version TEXT NOT NULL,
                    campaign_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, actor_role TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS memory_exposures ON memory_accesses(exposure_id);
                CREATE INDEX IF NOT EXISTS memory_version_accesses ON memory_accesses(version);
                CREATE TABLE IF NOT EXISTS memory_judgments (
                    exposure_id TEXT PRIMARY KEY, judgment TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_lifecycle (
                    event_id TEXT PRIMARY KEY, version TEXT NOT NULL,
                    event TEXT NOT NULL);
            ''')

    @classmethod
    def for_kdb(cls, path):
        return cls(str(Path(path).resolve()) + '.memory.sqlite3')

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _insert_record(self, db, record, status):
        from .subject_knowledge import memory_artifacts
        version = digest(record)
        db.execute('INSERT INTO memory_versions VALUES (?,?,?,?)',
                   (version, record['memory_id'], _json(record), status))
        db.executemany('INSERT INTO memory_artifacts VALUES (?,?)',
                       [(version, sha) for sha in {version, *memory_artifacts(record)}])
        for base in record['proposal'].get('base_versions', []):
            db.execute('INSERT INTO memory_lineage VALUES (?,?)', (version, base['version']))
        return version

    def bootstrap(self, record):
        """Register an existing accepted local head without overriding shared history."""
        version = digest(record)
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM memory_versions WHERE version=?', (version,)).fetchone():
                return version
            head = db.execute('SELECT version FROM memory_heads WHERE memory_id=?', (record['memory_id'],)).fetchone()
            self._insert_record(db, record, 'superseded' if head else 'active')
            if head is None:
                db.execute('INSERT INTO memory_heads VALUES (?,?)', (record['memory_id'], version))
        return version

    @staticmethod
    def _bases_current(db, bases):
        return all(db.execute('SELECT 1 FROM memory_heads WHERE memory_id=? AND version=?',
                              (base['memory_id'], base['version'])).fetchone() for base in bases)

    def bases_current(self, bases):
        with self._transaction() as db:
            return self._bases_current(db, bases)

    def records(self):
        with self._transaction() as db:
            rows = db.execute("SELECT v.record FROM memory_heads h JOIN memory_versions v USING(version) "
                              "WHERE v.status='active' ORDER BY h.memory_id").fetchall()
        return [json.loads(row['record']) for row in rows]

    def publish(self, record, *, expected_statuses=None):
        """Compare all shared heads and publish one replacement atomically."""
        version = digest(record)
        bases = record['proposal'].get('base_versions', [])
        if len({base['memory_id'] for base in bases}) != len(bases):
            raise ValueError('memory bases must identify distinct memories')
        with self._transaction() as db:
            previous = db.execute('SELECT record FROM memory_versions WHERE version=?', (version,)).fetchone()
            if previous:
                if previous['record'] != _json(record):
                    raise ValueError('immutable memory record changed')
                return version
            if not self._bases_current(db, bases):
                raise StaleMemory('shared memory replacement base changed; reconcile source review')
            if record['memory_id'] not in {base['memory_id'] for base in bases} and db.execute(
                    'SELECT 1 FROM memory_heads WHERE memory_id=?', (record['memory_id'],)).fetchone():
                raise StaleMemory('shared memory identity already exists; propose an explicit replacement')
            resolutions = {row['tombstone_id']: row for row in record['provenance'].get('retirement_resolutions', [])}
            for base in bases:
                status = db.execute('SELECT status FROM memory_versions WHERE version=?', (base['version'],)).fetchone()
                if status is None or status['status'] not in {'active', 'quarantined', 'retired'}:
                    raise StaleMemory('shared memory base is no longer current')
                if expected_statuses is not None and status['status'] != expected_statuses[base['version']]:
                    raise StaleMemory('shared memory lifecycle changed during publication; repeat source review')
                if status['status'] == 'retired':
                    events = db.execute('SELECT event FROM memory_lifecycle WHERE version=?', (base['version'],)).fetchall()
                    for row in events:
                        event = json.loads(row['event'])
                        if event['status'] != 'retired':
                            continue
                        resolution = resolutions.get(event['event_id'])
                        if (not resolution or not resolution.get('reason') or not
                                set(_evidence(resolution['evidence'])) - set(event['evidence'])):
                            raise ValueError('retired memory replacement requires new source-reviewed resolving evidence')
            self._insert_record(db, record, 'active')
            for base in bases:
                db.execute('DELETE FROM memory_heads WHERE memory_id=?', (base['memory_id'],))
                db.execute("UPDATE memory_versions SET status='superseded' WHERE version=? AND status='active'", (base['version'],))
            db.execute('INSERT INTO memory_heads VALUES (?,?)', (record['memory_id'], version))
        return version

    def replacements(self, version):
        """Published active children prove resolution without reviving the old text."""
        with self._transaction() as db:
            rows = db.execute('SELECT v.memory_id,v.version,v.status,v.record FROM memory_versions v '
                "JOIN memory_lineage l ON v.version=l.child WHERE l.parent=? AND v.status='active' ORDER BY v.version",
                (version,)).fetchall()
        return [dict(row, record=json.loads(row['record'])) for row in rows]

    def record(self, version):
        with self._transaction() as db:
            row = db.execute('SELECT record FROM memory_versions WHERE version=?', (version,)).fetchone()
        if row is None:
            raise KeyError(version)
        return json.loads(row['record'])

    def status(self, version):
        with self._transaction() as db:
            row = db.execute('SELECT status FROM memory_versions WHERE version=?', (version,)).fetchone()
        return row['status'] if row else None

    def available(self, version):
        return self.status(version) in {'active', 'superseded'}

    def access(self, event_id, *, memory_id, version, campaign_id, project_id, actor_id, actor_role):
        """Record actual delivery, idempotently by the host's durable tool event."""
        if not all(isinstance(v, str) and v for v in (event_id, memory_id, version, campaign_id, project_id, actor_id, actor_role)):
            raise ValueError('memory delivery requires explicit actor/project/event identities')
        identity = dict(memory_id=memory_id, version=version, campaign_id=campaign_id,
                        project_id=project_id, actor_id=actor_id, actor_role=actor_role)
        exposure = digest(identity)
        with self._transaction() as db:
            old = db.execute('SELECT * FROM memory_accesses WHERE event_id=?', (event_id,)).fetchone()
            row = dict(event_id=event_id, exposure_id=exposure, **identity)
            if old:
                if dict(old) != row:
                    raise ValueError('memory access event reused for a different delivery')
                return exposure
            current = db.execute('SELECT memory_id,status FROM memory_versions WHERE version=?', (version,)).fetchone()
            if current is None or current['memory_id'] != memory_id or current['status'] not in {'active', 'superseded'}:
                raise ValueError('memory version is unavailable for actor delivery')
            db.execute('INSERT INTO memory_accesses VALUES (?,?,?,?,?,?,?,?)', tuple(row.values()))
        return exposure

    def exposures(self, campaign_id, project_id):
        with self._transaction() as db:
            rows = db.execute('''SELECT a.exposure_id,a.memory_id,a.version,a.campaign_id,
                a.project_id,a.actor_id,a.actor_role,COUNT(*) AS accesses,v.record,j.judgment
                FROM memory_accesses a JOIN memory_versions v USING(version)
                LEFT JOIN memory_judgments j USING(exposure_id)
                WHERE a.campaign_id=? AND a.project_id=? GROUP BY a.exposure_id
                ORDER BY a.exposure_id''', (campaign_id, project_id)).fetchall()
        return [dict(dict(row), record=json.loads(row['record']),
                     judgment=json.loads(row['judgment']) if row['judgment'] else None) for row in rows]

    def judge(self, exposure_id, *, usefulness, correctness, reason, evidence):
        if usefulness not in {'useful', 'not-useful', 'unknown'} or correctness not in {'supported', 'contradicted', 'unknown'}:
            raise ValueError('invalid memory judgment')
        evidence = _evidence(evidence)
        if not isinstance(reason, str) or not reason.strip() or correctness == 'contradicted' and not evidence:
            raise ValueError('memory judgment needs a reason and contradiction evidence')
        judgment = dict(usefulness=usefulness, correctness=correctness, reason=reason, evidence=evidence)
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM memory_accesses WHERE exposure_id=?', (exposure_id,)).fetchone() is None:
                raise KeyError(exposure_id)
            old = db.execute('SELECT judgment FROM memory_judgments WHERE exposure_id=?', (exposure_id,)).fetchone()
            if old and old['judgment'] != _json(judgment):
                raise ValueError('memory exposure already has a different judgment')
            db.execute('INSERT OR IGNORE INTO memory_judgments VALUES (?,?)', (exposure_id, _json(judgment)))
        return judgment

    def stats(self, memory_id, version=None):
        """Aggregate lineage by distinct event/exposure IDs, never copied totals."""
        with self._transaction() as db:
            if version is None:
                versions = {row[0] for row in db.execute('SELECT version FROM memory_versions WHERE memory_id=?', (memory_id,))}
                pending = list(versions)
                while pending:
                    for row in db.execute('SELECT parent FROM memory_lineage WHERE child=?', (pending.pop(),)):
                        if row[0] not in versions:
                            versions.add(row[0]); pending.append(row[0])
            else:
                versions = {version}
            placeholders = ','.join('?' for _ in versions)
            rows = [dict(row) for row in db.execute('SELECT a.*,j.judgment FROM memory_accesses a '
                'LEFT JOIN memory_judgments j USING(exposure_id) WHERE a.version IN (' + placeholders + ')', tuple(versions))] if versions else []
        def aggregate(items):
            exposures = {row['exposure_id']: json.loads(row['judgment']) if row['judgment'] else None for row in items}
            useful = sum(bool(j) and j['usefulness'] == 'useful' for j in exposures.values())
            not_useful = sum(bool(j) and j['usefulness'] == 'not-useful' for j in exposures.values())
            decided = useful + not_useful
            return dict(accesses=len(items), exposures=len(exposures), reviewed=sum(j is not None for j in exposures.values()),
                        unreviewed=sum(j is None for j in exposures.values()), useful=useful, not_useful=not_useful,
                        unknown=len(exposures)-decided, decided=decided,
                        usefulness_rate=useful/decided if decided else None,
                        hot=len(items) >= 10, reviewed_hot=decided >= 10,
                        needs_rewrite=decided >= 10 and useful/decided < .3)
        by_actor = {}
        for row in rows:
            key = _json([row['campaign_id'], row['actor_id'], row['actor_role']])
            by_actor.setdefault(key, []).append(row)
        return dict(aggregate(rows), by_actor={key: dict(actor_id=items[0]['actor_id'], actor_role=items[0]['actor_role'],
            campaign_id=items[0]['campaign_id'], **aggregate(items)) for key, items in by_actor.items()})

    def transition(self, version, status, *, event_id, reason, evidence, expected_status=None):
        if status not in {'active', 'quarantined', 'retired'} or not isinstance(reason, str) or not reason.strip():
            raise ValueError('invalid memory lifecycle transition')
        evidence = _evidence(evidence)
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('memory lifecycle event requires an identity')
        if not evidence:
            raise ValueError('memory lifecycle transitions require evidence')
        event = dict(event_id=event_id, version=version, status=status, reason=reason, evidence=evidence)
        with self._transaction() as db:
            old = db.execute('SELECT event FROM memory_lifecycle WHERE event_id=?', (event_id,)).fetchone()
            if old:
                if old['event'] != _json(event):
                    raise ValueError('memory lifecycle event reused')
                return event
            row = db.execute('SELECT status FROM memory_versions WHERE version=?', (version,)).fetchone()
            if row is None:
                raise KeyError(version)
            if expected_status is not None and row['status'] != expected_status:
                raise ValueError('memory lifecycle changed; reconcile current version')
            if status == 'active' and not db.execute(
                    'SELECT 1 FROM memory_heads WHERE version=?', (version,)).fetchone():
                raise ValueError('superseded memory cannot become the current version again')
            db.execute('INSERT INTO memory_lifecycle VALUES (?,?,?)', (event_id, version, _json(event)))
            db.execute('UPDATE memory_versions SET status=? WHERE version=?', (status, version))
        return event

    def retired(self, query=''):
        words = set(re.findall(r'\w+', query.casefold()))
        with self._transaction() as db:
            rows = db.execute('SELECT l.event,v.memory_id,v.record FROM memory_lifecycle l JOIN memory_versions v USING(version)').fetchall()
        result = []
        for row in rows:
            event = json.loads(row['event'])
            if event['status'] != 'retired' or words and not words & set(re.findall(r'\w+', row['record'].casefold())):
                continue
            result.append(dict(tombstone_id=event['event_id'], memory_id=row['memory_id'], version=event['version'],
                record=json.loads(row['record']), reason=event['reason'], evidence=event['evidence']))
        return result

    def artifact_versions(self, sha):
        with self._transaction() as db:
            return [dict(row) for row in db.execute(
                'SELECT v.memory_id,v.version,v.status FROM memory_versions v '
                'JOIN memory_artifacts a USING(version) WHERE a.sha=? ORDER BY v.version', (sha,))]

    def require_actor_artifact(self, sha, version=None):
        """Guard knowledge-only reads; callers retain project/reference authority."""
        with self._transaction() as db:
            rows = db.execute('''SELECT v.version,v.status FROM memory_versions v
                JOIN memory_artifacts a USING(version) WHERE a.sha=?''', (sha,)).fetchall()
        if version is not None:
            rows = [row for row in rows if row['version'] == version]
            if not rows:
                raise ValueError('artifact does not belong to the selected memory version')
        if rows and all(row['status'] in {'quarantined', 'retired'} for row in rows):
            raise ValueError('memory artifact withdrawn pending evidence-backed resolution')

    def report(self):
        with self._transaction() as db:
            rows = db.execute('SELECT version,memory_id,status FROM memory_versions ORDER BY memory_id,version').fetchall()
        return [dict(row, current=self.stats(row['memory_id'], row['version']),
                     lifetime=self.stats(row['memory_id'])) for row in rows]
