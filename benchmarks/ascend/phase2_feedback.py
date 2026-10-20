"""Phase 2 observations, not verified knowledge; imports never execute workloads."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from optimizer.jobs import JobStore, atomic, canonical, sha
from optimizer.transport import exclusive


STATUSES = {'open', 'investigating', 'ready-to-retry', 'resolved', 'deferred'}


def _read(path, default=None):
    return json.loads(path.read_bytes()) if path.is_file() else default


def _category(task, result):
    if result.get('verified') is True:
        speedup = result.get('speedup')
        goal = task.get('performance_goal', {})
        threshold = goal.get('speedup', 1.0) if isinstance(goal, dict) else goal
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            threshold = 1.0
        measured = (isinstance(speedup, (int, float)) and not isinstance(speedup, bool)
                    and math.isfinite(speedup) and speedup > 0)
        return ('verified-slow' if speedup < threshold else 'verified-success') if measured else 'verified-timing-unknown'
    status = str((result.get('result') or {}).get('status', result.get('classification', '')))
    if 'compile' in status:
        return 'compile-failure'
    if 'budget' in status:
        return 'budget-exhausted'
    return 'incorrect' if result.get('candidate') or result.get('source_sha256') else 'no-candidate'


class FeedbackStore:
    """Immutable outcomes with separately editable investigation state.

    Source roots and artifact hashes are provenance, not portable embedded files.
    Missing artifacts are reported; imports never fetch remote profiler reports.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def records(self):
        return [_read(path) for path in sorted(self.root.glob('*.json'))]

    def update_investigation(self, feedback_id, **fields):
        if set(fields) - {'status', 'classification', 'reason', 'projects', 'retries', 'evidence'}:
            raise ValueError('unknown investigation field')
        with exclusive(self.root / '.lock', wait=True):
            path = self.root / (feedback_id + '.json')
            record = _read(path)
            if record is None:
                raise KeyError(feedback_id)
            value = {**record['investigation'], **fields}
            if value['status'] not in STATUSES:
                raise ValueError('invalid investigation status')
            if value['status'] == 'deferred' and not value.get('reason'):
                raise ValueError('deferral requires a reason')
            if value['status'] == 'ready-to-retry' and not value.get('evidence'):
                raise ValueError('ready-to-retry requires investigation evidence')
            control = record['outcome']['category'] in {'verified-success', 'verified-timing-unknown'}
            if (value['status'] == 'resolved' and not control
                    and not value.get('retries')):
                raise ValueError('resolution requires a linked successful original-task retry')
            if value['status'] == 'resolved' and not control:
                retries = [_read(self.root / (key + '.json'), {}) for key in value['retries']]
                if not any(r.get('outcome', {}).get('verified') is True
                           and r['outcome']['task'] == record['outcome']['task']
                           and r['outcome']['target'] == record['outcome']['target']
                           and r['outcome']['language'] == record['outcome']['language']
                           and (record['outcome']['category'] != 'verified-slow'
                                or _category(record['outcome']['task'], r['outcome']['result']) == 'verified-success')
                           for r in retries):
                    raise ValueError('resolution requires a verified successful original-task retry')
            record['investigation'] = value
            atomic(path, canonical(record))
            return record

    def _put(self, outcome):
        identity = {k: outcome[k] for k in ('source_root', 'attempt_id', 'completed')}
        key = sha(canonical(identity))
        path = self.root / (key + '.json')
        with exclusive(self.root / '.lock', wait=True):
            if path.exists():
                return False
            record = dict(feedback_id=key, outcome=outcome,
                investigation=dict(status='resolved' if outcome['category'] in
                    {'verified-success', 'verified-timing-unknown'} else 'open',
                    classification=None, reason=None, projects=[], retries=[], evidence=[]))
            atomic(path, canonical(record))
        return True

    def import_campaign(self, root):
        root = Path(root).resolve()
        report = dict(source_root=str(root), imported=0, existing=0, warnings=[])
        if not root.is_dir():
            report['warnings'].append('source root is missing')
            return report
        paths = [root / 'state.json'] if (root / 'state.json').exists() else sorted(root.rglob('state.json'))
        supported = 0
        for path in paths:
            try:
                state = _read(path)
            except (ValueError, OSError) as error:
                report['warnings'].append(f'{path}: unreadable state: {error}')
                continue
            if state.get('phase') != 'optimizer-phase2':
                continue
            supported += 1
            campaign = path.parent
            metadata = _read(campaign / 'campaign-identity.json', {})
            seed = _read(campaign / 'seed-memory.json', {})
            metadata = {**seed.get('source_identity', {}), **metadata}
            rows = {row['attempt_id']: row for row in state.get('attempts', [])}
            for retained in sorted((campaign / 'attempts').glob('*.json')):
                row = _read(retained)
                rows.setdefault(row['attempt_id'], row)
            for row in rows.values():
                result = row['result']
                outcome = self._outcome(campaign, row, metadata, report)
                outcome.update(category=_category(row['task'], result), completed=True)
                report['imported' if self._put(outcome) else 'existing'] += 1
            pending = _read(campaign / 'pending.json')
            if pending and pending['attempt_id'] not in rows:
                relative = 'tasks/' + pending['task']['task_id'] + '-' + sha(canonical(pending['task']))[:12]
                jobs = campaign / relative / 'jobs'
                manifests = sorted(jobs.glob('*/manifest.json'))
                # A pending selection is not evidence of an interrupted execution.
                for manifest in manifests:
                    job = _read(manifest)
                    if job.get('status') not in {'failed', 'stopped', 'budget-exhausted'}:
                        continue
                    row = {**pending, 'execution_directory': relative,
                           'result': dict(job_id=job['job_id'], verified=False,
                               classification=job['status'], handles=job.get('handles', []))}
                    outcome = self._outcome(campaign, row, metadata, report)
                    outcome.update(completed=False, category='budget-exhausted' if job['status'] == 'budget-exhausted'
                                   else 'interrupted', job_status=job)
                    report['imported' if self._put(outcome) else 'existing'] += 1
        if not supported:
            report['warnings'].append('no supported optimizer Phase 2 state found; legacy layouts are not imported')
        return report

    def _outcome(self, campaign, row, metadata, report):
        result = row['result']
        directory = campaign / row.get('execution_directory', '')
        runtime = {}
        job_id = result.get('job_id')
        if job_id and (directory / 'jobs' / job_id / 'events.jsonl').is_file():
            try:
                events = JobStore(directory / 'jobs').events(job_id)
                runtime = next((e['payload']['runtime'] for e in events if e['kind'] == 'phase2-task'), {})
            except (ValueError, OSError, KeyError) as error:
                report['warnings'].append(f'{directory}: runtime evidence unreadable: {error}')
        evidence = {}
        for role in ('candidate', 'baseline', 'profiling', 'result_artifact'):
            ref = result.get(role)
            if not ref:
                continue
            evidence[role] = dict(ref, job_root=str(directory / 'jobs'), job_id=job_id)
            try:
                artifact = JobStore(directory / 'jobs').artifact_path(job_id, ref['sha256'])
                with artifact.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != ref['sha256']:
                        raise ValueError('artifact digest mismatch')
            except (ValueError, OSError, KeyError, TypeError) as error:
                report['warnings'].append(f'{campaign}:{row["attempt_id"]}: missing {role} evidence: {error}')
        return dict(source_root=str(campaign), attempt_id=row['attempt_id'],
            kind=row.get('kind', 'target'), target=runtime.get('target', metadata.get('target', 'unknown')),
            language=runtime.get('language', metadata.get('language',
                'ascend-c' if runtime.get('target') == 'a3' else 'unknown')),
            runtime=runtime, task=row['task'], verified=result.get('verified') is True,
            result=result, evidence=evidence, interpretation='Observation, not verified reusable knowledge')
