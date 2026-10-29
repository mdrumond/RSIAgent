"""Evidence-driven maintenance obligations projected into ordinary curriculum."""
from dataclasses import replace
import json
import sqlite3
from pathlib import Path

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts

from optimizer.jobs import canonical
from .subject_memory import digest


class MemoryMaintenance:
    def __init__(self, campaign, ledger):
        self.campaign, self.ledger = campaign, ledger
        self.campaign_id = str(campaign.root.resolve())
        self.shared = ContentAddressedArtifacts(Path(str(ledger.path) + '.artifacts'))
        with sqlite3.connect(ledger.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS memory_maintenance (
                repair_id TEXT PRIMARY KEY, version TEXT NOT NULL, kind TEXT NOT NULL,
                payload TEXT NOT NULL, resolution TEXT)''')
            db.execute('''CREATE UNIQUE INDEX IF NOT EXISTS open_memory_maintenance
                ON memory_maintenance(version,kind) WHERE resolution IS NULL''')
            db.execute('''CREATE TABLE IF NOT EXISTS memory_maintenance_triggers (
                trigger_id TEXT PRIMARY KEY, repair_id TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS memory_maintenance_reviews (
                exposure_id TEXT PRIMARY KEY, payload TEXT NOT NULL, applied INTEGER NOT NULL)''')

    def read_artifact(self, sha, *, remaining_seconds):
        if remaining_seconds <= 0:
            raise TimeoutError('memory artifact read allocation exhausted')
        try:
            return self.campaign.artifacts.read_bytes(sha)
        except FileNotFoundError:
            data = self.shared.read_bytes(sha)
            if self.campaign.artifacts.put_bytes(data) != sha:
                raise ValueError('shared memory artifact identity changed')
            return data

    def retain(self, sha):
        data = self.read_artifact(sha, remaining_seconds=float('inf'))
        if self.shared.put_bytes(data) != sha:
            raise ValueError('retained memory artifact identity changed')

    def archive(self, record):
        from .subject_knowledge import memory_artifacts
        self.shared.put_bytes(canonical(record))
        for sha in memory_artifacts(record):
            self.retain(sha)

    def obligations(self, *, open_only=True):
        with sqlite3.connect(self.ledger.path) as db:
            rows = db.execute('SELECT payload,resolution FROM memory_maintenance ORDER BY repair_id').fetchall()
        return [dict(json.loads(payload), resolution=json.loads(resolution) if resolution else None)
                for payload, resolution in rows if not open_only or resolution is None]

    def _ensure(self, version, kind, *, project_id, subject_id, reason, evidence, trigger):
        # A new counterexample after restoration creates a new episode. Replaying
        # an old assessment returns its original episode, even after resolution.
        memory_id = self.ledger.record(version)['memory_id']
        with sqlite3.connect(self.ledger.path) as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM memory_maintenance_triggers WHERE trigger_id=?', (trigger,)).fetchone():
                return
            old = db.execute('SELECT repair_id FROM memory_maintenance WHERE version=? AND kind=? AND resolution IS NULL',
                             (version, kind)).fetchone()
            repair_id = old[0] if old else 'memory-' + kind + '-' + version + '-' + digest(trigger)[:12]
            if old is None:
                payload = dict(repair_id=repair_id, memory_id=memory_id,
                    version=version, kind=kind, project_id=project_id, subject_id=subject_id,
                    reason=reason, evidence=sorted(set(evidence)))
                db.execute('INSERT INTO memory_maintenance VALUES (?,?,?,?,NULL)',
                           (repair_id, version, kind, canonical(payload).decode()))
            db.execute('INSERT INTO memory_maintenance_triggers VALUES (?,?)', (trigger, repair_id))

    def project_obligations(self):
        """Any campaign can resume shared work; an abandoned owner holds no lease."""
        scheduler = self.campaign.scheduler
        for row in self.obligations():
            subject = row['subject_id'] if row['subject_id'] in scheduler.index else scheduler.subjects[0].subject_id
            scheduler.add_repair(row['repair_id'], subject, row['project_id'],
                row['reason'] + ' Resolve memory maintenance explicitly: ' + json.dumps(dict(
                    kind=row['kind'], memory_id=row['memory_id'], version=row['version']), sort_keys=True),
                row['evidence'])

    def prepare(self, request, repair_ids=()):
        def document(row):
            record = row.pop('record', None) or self.ledger.record(row['version'])
            from .subject_knowledge import memory_artifacts
            for sha in {*memory_artifacts(record), *row.get('evidence', [])}:
                self.read_artifact(sha, remaining_seconds=float('inf'))
            return dict(row, artifact_sha256=sorted(memory_artifacts(record)), status=self.ledger.status(row['version']),
                claim_identity=[' '.join(str(record['proposal'].get(key, '')).casefold().split()) for key in
                    ('summary','applicability','preconditions','expectations','limitations')],
                document_sha256=self.campaign.artifacts.put_bytes(canonical(record)))
        exposures = [document(dict(row)) for row in self.ledger.exposures(self.campaign_id, request.project_id) if row['judgment'] is None]
        for row in exposures:
            row['statistics'] = {key:value for key,value in self.ledger.stats(row['memory_id'], row['version']).items() if key != 'by_actor'}
            row['lifetime_statistics'] = {key:value for key,value in self.ledger.stats(row['memory_id']).items() if key != 'by_actor'}
        repairs = []
        for obligation in self.obligations():
            if obligation['repair_id'] not in repair_ids:
                continue
            with sqlite3.connect(self.ledger.path) as db:
                triggers = db.execute('SELECT trigger_id FROM memory_maintenance_triggers WHERE repair_id=?',
                                      (obligation['repair_id'],)).fetchall()
                assessments = [db.execute('SELECT payload FROM memory_maintenance_reviews WHERE exposure_id=?',
                               (trigger[0].split(':',1)[1],)).fetchone() for trigger in triggers]
            evidence = set(obligation['evidence'])
            for assessment in assessments:
                if assessment:
                    evidence.update(json.loads(assessment[0])['judgment']['evidence'])
            repairs.append(document(dict(obligation, evidence=sorted(evidence))))
        # All retired claim metadata is visible; exact documents/evidence are paged by hash.
        retired = [document(dict(row)) for row in self.ledger.retired()]
        artifacts = dict(request.artifacts)
        if repairs:
            old = self.ledger.record(repairs[0]['version'])['proposal']['after']['sha256']
            if old != request.source_sha256:
                artifacts.setdefault('original-source', old)
        return replace(request, artifacts=artifacts, memory_exposures=tuple(exposures), memory_repairs=tuple(repairs),
                       retired_memories=tuple(retired))

    def observe(self, project_id, record):
        """Replay saved reviews to recover judge/quarantine/enqueue crash gaps."""
        review = record.get('review')
        if review is None or not review.get('memory_judgments'):
            return
        subject = next((s.subject_id for s in self.campaign.scheduler.subjects
                        if s.objective == record['objective']), self.campaign.scheduler.subjects[0].subject_id)
        exposures = {row['exposure_id']: row for row in self.ledger.exposures(self.campaign_id, project_id)}
        for judgment in review.get('memory_judgments', []):
            row = exposures[judgment['exposure_id']]
            self.archive(row['record'])
            for sha in judgment['evidence']:
                self.retain(sha)
            self.retain(record['review_sha256'])
            payload = dict(exposure=row, judgment=judgment, project_id=project_id,
                           subject_id=subject, review_sha256=record['review_sha256'])
            with sqlite3.connect(self.ledger.path) as db:
                previous = db.execute('SELECT payload FROM memory_maintenance_reviews WHERE exposure_id=?',
                                      (judgment['exposure_id'],)).fetchone()
                if previous and json.loads(previous[0]) != payload:
                    # Delivery counters can grow, but the verifier's decision is immutable.
                    if json.loads(previous[0])['judgment'] != judgment:
                        raise ValueError('memory exposure already has a different assessment')
                db.execute('INSERT OR IGNORE INTO memory_maintenance_reviews VALUES (?,?,0)',
                           (judgment['exposure_id'], canonical(payload).decode()))
        self._apply_pending()
        self.project_obligations()

    def _apply_pending(self):
        with sqlite3.connect(self.ledger.path) as db:
            pending = db.execute('SELECT exposure_id,payload FROM memory_maintenance_reviews WHERE applied=0').fetchall()
        for exposure_id, raw in pending:
            payload = json.loads(raw)
            row, judgment = payload['exposure'], payload['judgment']
            self.ledger.judge(exposure_id, **{key: judgment[key] for key in
                ('usefulness', 'correctness', 'reason', 'evidence')})
            version = row['version']
            kind = 'correct' if judgment['correctness'] == 'contradicted' else 'investigate' if judgment['investigate'] else None
            if kind == 'correct' and self.ledger.status(version) == 'retired':
                kind = None  # Already withdrawn permanently; retain the new judgment only.
            if kind == 'correct':
                self.ledger.transition(version, 'quarantined', event_id='contradiction-' + exposure_id,
                    reason=judgment['reason'], evidence=judgment['evidence'])
            if kind:
                self._ensure(version, kind, project_id=payload['project_id'], subject_id=payload['subject_id'],
                    reason=judgment['reason'], evidence=judgment['evidence'], trigger=kind + ':' + exposure_id)
            with sqlite3.connect(self.ledger.path) as db:
                db.execute('UPDATE memory_maintenance_reviews SET applied=1 WHERE exposure_id=?', (exposure_id,))
        for row in self.ledger.report():
            stats = row['current']
            if row['status'] == 'active' and stats['needs_rewrite']:
                proof = dict(memory_id=row['memory_id'], version=row['version'],
                             statistics={key:value for key,value in stats.items() if key != 'by_actor'})
                evidence = self.shared.put_json(proof)
                self._ensure(row['version'], 'rewrite', project_id='memory-statistics-' + row['version'],
                    subject_id=self.campaign.scheduler.subjects[0].subject_id,
                    reason='Frequently accessed current version is useful in fewer than 30% of decided exposures.',
                    evidence=[evidence], trigger='rewrite:' + row['version'])

    def reconcile(self):
        self._apply_pending()
        for project_id, record in self.campaign.state['projects'].items():
            self.observe(project_id, record)
        self.project_obligations()
        # Global resolution may have been completed by a different campaign.
        local = self.campaign.scheduler.state()['repairs']
        for row in self.obligations(open_only=False):
            if row['resolution'] is not None and row['repair_id'] in local and local[row['repair_id']]['status'] == 'open':
                self.campaign.scheduler.resolve_repair(row['repair_id'], row['resolution']['evidence'])

    def prioritize(self):
        open_rows = self.obligations()
        if not open_rows:
            return None
        order = {'correct': 0, 'investigate': 1, 'rewrite': 2}
        row = min(open_rows, key=lambda row: (order[row['kind']], row['repair_id']))
        local = self.campaign.scheduler.state()['repairs'][row['repair_id']]
        self.campaign.scheduler.decide(dict(action='repair', repair_id=row['repair_id'],
            reason='Memory maintenance precedes ordinary curriculum work', evidence=row['evidence']))
        return local

    def resolve(self, project_id, record):
        """Only explicit reviewed resolutions close obligations after publication."""
        if record['review']['decision'] != 'accept':
            return
        obligations = {row['repair_id']: row for row in self.obligations(open_only=False)}
        for resolution in record['review'].get('memory_resolutions', []):
            row = obligations[resolution['repair_id']]
            outcome = dict(resolution, project_id=project_id, evidence=sorted(set(resolution['evidence'])))
            with sqlite3.connect(self.ledger.path) as db:
                settled = db.execute('SELECT resolution FROM memory_maintenance WHERE repair_id=?', (row['repair_id'],)).fetchone()[0]
                if settled:
                    continue
            status = self.ledger.status(row['version'])
            if resolution['action'] == 'replace':
                if not self.ledger.replacements(row['version']):
                    raise ValueError('memory correction must publish its replacement before resolving repair')
            else:
                reviewed = next(item for item in record['review_request']['memory_repairs'] if item['repair_id'] == row['repair_id'])
                event_id = 'resolution-' + row['repair_id'] + '-' + record['review_sha256']
                with sqlite3.connect(self.ledger.path) as db:
                    receipt = db.execute('SELECT 1 FROM memory_lifecycle WHERE event_id=?', (event_id,)).fetchone()
                if not receipt and resolution['action'] == 'restore' and (status != 'quarantined' or self.ledger.replacements(row['version'])):
                    raise ValueError('only a quarantined current version may be restored')
                wanted = 'retired' if resolution['action'] == 'retire' else 'active'
                self.ledger.transition(row['version'], wanted, event_id=event_id,
                    reason=resolution['reason'], evidence=resolution['evidence'], expected_status=reviewed['status'])
            for sha in resolution['evidence']:
                self.retain(sha)
            with sqlite3.connect(self.ledger.path) as db:
                # An explicit correction of this version also settles older maintenance
                # reasons for the same bytes; the replacement starts fresh counters.
                if resolution['action'] in {'replace', 'retire'}:
                    db.execute('UPDATE memory_maintenance SET resolution=? WHERE version=? AND resolution IS NULL',
                               (canonical(outcome).decode(), row['version']))
                else:
                    db.execute('UPDATE memory_maintenance SET resolution=? WHERE repair_id=? AND resolution IS NULL',
                               (canonical(outcome).decode(), row['repair_id']))
        self.reconcile()

    def report(self):
        return dict(memories=self.ledger.report(), obligations=self.obligations(open_only=False),
            retirements=[{key:row[key] for key in ('tombstone_id','memory_id','version','reason','evidence')}
                         for row in self.ledger.retired()])
