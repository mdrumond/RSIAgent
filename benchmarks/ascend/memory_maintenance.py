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
                payload TEXT NOT NULL, resolution TEXT, UNIQUE(version,kind))''')

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

    def _ensure(self, version, kind, *, project_id, subject_id, reason, evidence):
        repair_id = 'memory-' + kind + '-' + version
        payload = dict(repair_id=repair_id, memory_id=self.ledger.record(version)['memory_id'],
            version=version, kind=kind, project_id=project_id, subject_id=subject_id,
            reason=reason, evidence=sorted(set(evidence)))
        with sqlite3.connect(self.ledger.path) as db:
            db.execute('INSERT OR IGNORE INTO memory_maintenance VALUES (?,?,?,?,NULL)',
                       (repair_id, version, kind, canonical(payload).decode()))

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
        exposures = [document(dict(row)) for row in self.ledger.exposures(self.campaign_id, request.project_id)]
        for row in exposures:
            row['statistics'] = {key:value for key,value in self.ledger.stats(row['memory_id'], row['version']).items() if key != 'by_actor'}
            row['lifetime_statistics'] = {key:value for key,value in self.ledger.stats(row['memory_id']).items() if key != 'by_actor'}
        repairs = [document(dict(row)) for row in self.obligations() if row['repair_id'] in repair_ids]
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
        if review is None:
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
            self.ledger.judge(judgment['exposure_id'], **{key: judgment[key] for key in
                ('usefulness', 'correctness', 'reason', 'evidence')})
            version = row['version']
            if judgment['correctness'] == 'contradicted':
                self.ledger.transition(version, 'quarantined', event_id='contradiction-' + judgment['exposure_id'],
                    reason=judgment['reason'], evidence=judgment['evidence'])
                self._ensure(version, 'correct', project_id=project_id, subject_id=subject,
                    reason=judgment['reason'], evidence=judgment['evidence'])
            elif judgment.get('investigate'):
                self._ensure(version, 'investigate', project_id=project_id, subject_id=subject,
                    reason=judgment['reason'], evidence=judgment['evidence'])
            stats = self.ledger.stats(row['memory_id'], version)
            if stats['needs_rewrite'] and self.ledger.status(version) == 'active':
                self._ensure(version, 'rewrite', project_id=project_id, subject_id=subject,
                    reason='Frequently accessed current version is useful in fewer than 30% of decided exposures. ' + json.dumps(stats),
                    evidence=[record['review_sha256']])
        self.project_obligations()

    def reconcile(self):
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
                if status != reviewed['status']:
                    raise ValueError('memory lifecycle changed after review; reconcile its repair')
                if resolution['action'] == 'restore' and status != 'quarantined':
                    raise ValueError('only a quarantined current version may be restored')
                wanted = 'retired' if resolution['action'] == 'retire' else 'active'
                self.ledger.transition(row['version'], wanted,
                    event_id='resolution-' + row['repair_id'] + '-' + record['review_sha256'],
                    reason=resolution['reason'], evidence=resolution['evidence'], expected_status=status)
            for sha in resolution['evidence']:
                self.retain(sha)
            with sqlite3.connect(self.ledger.path) as db:
                # An explicit correction of this version also settles older maintenance
                # reasons for the same bytes; the replacement starts fresh counters.
                if resolution['action'] == 'replace':
                    db.execute('UPDATE memory_maintenance SET resolution=? WHERE version=? AND resolution IS NULL',
                               (canonical(outcome).decode(), row['version']))
                else:
                    db.execute('UPDATE memory_maintenance SET resolution=? WHERE repair_id=? AND resolution IS NULL',
                               (canonical(outcome).decode(), row['repair_id']))
        self.reconcile()

    def report(self):
        return dict(memories=self.ledger.report(), obligations=self.obligations(open_only=False))
