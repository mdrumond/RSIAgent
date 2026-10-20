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


def _observation_identity(outcome):
    identity = {key: outcome[key] for key in ('attempt_id', 'completed')}
    job_id = outcome['result'].get('job_id')
    identity['job_id' if job_id else 'source_root'] = job_id or outcome['source_root']
    return identity


def _logical_outcome(outcome):
    # Moving retained files changes locations, not the historical observation.
    return {**{key: value for key, value in outcome.items() if key not in {'source_root', 'evidence'}},
            'evidence': {role: {key: value for key, value in ref.items() if key != 'job_root'}
                         for role, ref in outcome['evidence'].items()}}


def _baseline_identity(outcome):
    """Read only retained, authenticated suite and comparison-policy evidence."""
    try:
        ref = outcome['evidence']['baseline']
        path = Path(ref['job_root']) / ref['job_id'] / 'artifacts' / ref['sha256']
        data = path.read_bytes()
        if sha(data) != ref['sha256']:
            return None
        value = json.loads(data)
        manifest = value.get('manifest', value)
        suite = value.get('suite_sha256', manifest.get('suite_sha256'))
        policy = manifest.get('policy')
        if isinstance(suite, str) and len(suite) == 64 and isinstance(policy, dict) and policy:
            return suite, policy
    except (KeyError, OSError, ValueError, TypeError):
        pass
    return None


def _verified_retry(outcome):
    from .optimizer_phase2_artifacts import verify_attempt_result
    try:
        ref = outcome['evidence']['result_artifact']
        if not Path(ref['job_root']).is_dir():
            return False
        verify_attempt_result(JobStore(ref['job_root']), outcome['result'],
                              attempt_id=outcome['attempt_id'], task_id=outcome['task']['task_id'])
        return outcome['result'].get('verified') is True
    except (KeyError, OSError, ValueError, TypeError):
        return False


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
    cases = (result.get('result') or {}).get('cases', [])
    compile_error = any(case.get('error') and not case.get('compilation') and
                        ('compiler_diagnostics' in case or 'compiler' in case['error']) for case in cases)
    if 'compile' in status or compile_error:
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

    def read_evidence(self, digest, offset=0, limit=8192):
        """Authenticate a locally retained linked artifact, then return one text page."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8192:
            raise ValueError('evidence page requires nonnegative offset and limit 1..8192')
        refs = [ref for record in self.records() for ref in record['outcome']['evidence'].values()
                if ref['sha256'] == digest]
        for ref in refs:
            try:
                if not Path(ref['job_root']).is_dir():
                    continue
                path = JobStore(ref['job_root']).artifact_path(ref['job_id'], digest)
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError('artifact digest mismatch')
                    size = path.stat().st_size
                    stream.seek(offset)
                    return dict(sha256=digest, offset=offset, size=size,
                                text=stream.read(limit).decode('utf-8', errors='replace'),
                                has_more=offset + limit < size,
                                next_offset=offset + limit if offset + limit < size else None)
            except FileNotFoundError:
                continue
        raise FileNotFoundError('linked evidence is not retained locally')

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
                if (record['outcome']['target'] not in {'a3', 'a5'}
                        or record['outcome']['language'] not in {'ascend-c', 'catlass-dsl'}):
                    raise ValueError('resolution requires known target and language identity')
                baseline = _baseline_identity(record['outcome'])
                if baseline is None:
                    raise ValueError('resolution requires authenticated frozen suite and comparison policy')
                known = {row['feedback_id']: row for row in self.records()}
                retries = [known.get(key, {}) for key in value['retries']]
                if not any(r.get('outcome', {}).get('verified') is True
                           and _verified_retry(r['outcome'])
                           and r['outcome']['task'] == record['outcome']['task']
                           and r['outcome']['target'] == record['outcome']['target']
                           and r['outcome']['language'] == record['outcome']['language']
                           and _baseline_identity(r['outcome']) == baseline
                           and (record['outcome']['category'] != 'verified-slow'
                                or _category(record['outcome']['task'], r['outcome']['result']) == 'verified-success')
                           for r in retries):
                    raise ValueError('resolution requires a verified successful original-task retry')
            record['investigation'] = value
            atomic(path, canonical(record))
            return record

    def _put(self, outcome, warnings):
        identity = _observation_identity(outcome)
        if not outcome['result'].get('job_id'):
            warnings.append(f'{outcome["source_root"]}:{outcome["attempt_id"]}: '
                            'no retained job identity; relocation deduplication unavailable')
        key = sha(canonical(identity))
        path = self.root / (key + '.json')
        with exclusive(self.root / '.lock', wait=True):
            # Keep IDs and investigation work from older path-keyed feedback files.
            if not path.exists():
                existing = next((row for row in self.records()
                                 if _observation_identity(row['outcome']) == identity), None)
                if existing is not None:
                    path = self.root / (existing['feedback_id'] + '.json')
            if path.exists():
                if _logical_outcome(_read(path)['outcome']) != _logical_outcome(outcome):
                    warnings.append(f'{outcome["source_root"]}:{outcome["attempt_id"]}: '
                                    'source outcome changed; immutable imported observation preserved')
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
        paths = sorted(root.rglob('state.json'))
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
            metadata = {**seed.get('source_identity', {}), **metadata,
                        **_read(campaign / 'campaign-run.json', {})}
            rows = {row['attempt_id']: row for row in state.get('attempts', [])}
            for retained in sorted((campaign / 'attempts').glob('*.json')):
                row = _read(retained)
                rows.setdefault(row['attempt_id'], row)
            for row in rows.values():
                result = row['result']
                outcome = self._outcome(campaign, row, metadata, report)
                outcome.update(category=_category(row['task'], result), completed=True)
                report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
            pending = _read(campaign / 'pending.json')
            if pending and pending['attempt_id'] not in rows:
                relative = 'tasks/' + pending['task']['task_id'] + '-' + sha(canonical(pending['task']))[:12]
                jobs = campaign / relative / 'jobs'
                manifests = sorted(jobs.glob('*/manifest.json'))
                # A pending selection is not evidence of an interrupted execution.
                for manifest in manifests:
                    try:
                        store = JobStore(jobs)
                        job = store.get(manifest.parent.name)
                        completed = next((event['payload'] for event in reversed(store.events(job['job_id']))
                            if event['kind'] == 'phase2-attempt-result'
                            and event['payload'].get('attempt_id') == pending['attempt_id']), None)
                    except (ValueError, OSError) as error:
                        report['warnings'].append(f'{manifest}: retained job unreadable: {error}')
                        continue
                    if completed is not None:
                        row = {**pending, 'execution_directory': relative, 'result': completed}
                        outcome = self._outcome(campaign, row, metadata, report)
                        outcome.update(completed=True, category=_category(pending['task'], completed))
                        report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
                        continue
                    if job.get('status') not in {'failed', 'stopped', 'budget-exhausted'}:
                        continue
                    row = {**pending, 'execution_directory': relative,
                           'result': dict(job_id=job['job_id'], verified=False,
                               classification=job['status'], handles=job.get('handles', []))}
                    outcome = self._outcome(campaign, row, metadata, report)
                    outcome.update(completed=False, category='budget-exhausted' if job['status'] == 'budget-exhausted'
                                   else 'interrupted', job_status=job)
                    report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
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
                if not (directory / 'jobs').is_dir():
                    raise FileNotFoundError('retained job directory is missing')
                artifact = JobStore(directory / 'jobs').artifact_path(job_id, ref['sha256'])
                with artifact.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != ref['sha256']:
                        raise ValueError('artifact digest mismatch')
            except (ValueError, OSError, KeyError, TypeError) as error:
                report['warnings'].append(f'{campaign}:{row["attempt_id"]}: missing {role} evidence: {error}')
        target = metadata.get('target')
        if target not in {'a3', 'a5'}:
            target = {'ascend-a3': 'a3', 'ascend-a5': 'a5'}.get(metadata.get('benchmark'), 'unknown')
        return dict(source_root=str(campaign), attempt_id=row['attempt_id'],
            kind=row.get('kind', 'target'), target=target,
            language=metadata.get('language', runtime.get('language', 'unknown')),
            runtime=runtime, task=row['task'], verified=result.get('verified') is True,
            result=result, evidence=evidence, interpretation='Observation, not verified reusable knowledge')
