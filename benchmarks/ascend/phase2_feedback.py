"""Failure observations, not verified knowledge; imports never execute workloads."""
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


def _read_import(path, warnings, default=None):
    try:
        value = _read(path, default)
        if value is not None and not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (ValueError, OSError) as error:
        warnings.append(f'{path}: unreadable retained record: {error}')
        return default


def _observation_identity(outcome):
    identity = {key: outcome[key] for key in ('attempt_id', 'completed')}
    job_id = outcome['result'].get('job_id')
    identity['job_id' if job_id else 'source_root'] = job_id or outcome['source_root']
    if job_id and outcome.get('job_incarnation'):
        identity['job_incarnation'] = outcome['job_incarnation']
    return identity


def _same_observation(old, new):
    previous, current = _observation_identity(old), _observation_identity(new)
    if not old.get('job_incarnation'):
        current.pop('job_incarnation', None)  # Older records cannot distinguish recreated jobs.
    return previous == current


def _logical_outcome(outcome):
    # Moving retained files changes locations, not the historical observation.
    return {**{key: value for key, value in outcome.items() if key not in {'source_root', 'evidence', 'job_incarnation', 'source_type', 'original_spec'}},
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
        submission = outcome['task']['submission']
        if 'signatures' not in submission and 'signature' in submission:
            submission = dict(submission, signatures=submission['signature']['signatures'])
        verify_attempt_result(JobStore(ref['job_root']), outcome['result'],
                              attempt_id=outcome['attempt_id'], task_id=outcome['task']['task_id'],
                              task_submission=submission)
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
        return [self._present(_read(path)) for path in sorted(self.root.glob('*.json'))]

    @staticmethod
    def _present(record):
        # Before generic feedback, every stored observation was Phase 2. Supply
        # additive dispatch metadata on reads without rewriting immutable history.
        outcome = record['outcome']
        outcome.setdefault('source_type', 'phase2')
        if outcome['source_type'] == 'phase2':
            outcome.setdefault('original_spec', outcome['task'])
        control = record['outcome']['category'] in {'verified-success', 'verified-timing-unknown'}
        record['lifecycle'] = None if control else {
            'resolved': 'fixed', 'ready-to-retry': 'remediated'
        }.get(record['investigation']['status'], 'failed')
        return record

    def read_evidence(self, digest, offset=0, limit=8192):
        """Authenticate a locally retained linked artifact, then return one text page."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8192:
            raise ValueError('evidence page requires nonnegative offset and limit 1..8192')
        refs = [ref for record in self.records() for ref in record['outcome']['evidence'].values()
                if ref['sha256'] == digest]
        for ref in refs:
            try:
                if ref.get('retained_path'):
                    path = Path(ref['retained_path'])
                else:
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

    def update_investigation(self, feedback_id, *, merge=None, **fields):
        """Optionally merge against current state under the same publication lock."""
        if set(fields) - {'status', 'classification', 'reason', 'projects', 'retries', 'evidence'}:
            raise ValueError('unknown investigation field')
        with exclusive(self.root / '.lock', wait=True):
            path = self.root / (feedback_id + '.json')
            record = _read(path)
            if record is None:
                raise KeyError(feedback_id)
            if merge is not None:
                merged = merge(record['investigation'])
                if merged is None:
                    return record
                fields = {**fields, **merged}
                if set(fields) - {'status', 'classification', 'reason', 'projects', 'retries', 'evidence'}:
                    raise ValueError('unknown investigation field')
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
            if value['status'] == 'resolved' and not control and record['outcome'].get('source_type') in {'phase1', 'optimizer'}:
                known = {row['feedback_id']: row for row in self.records()}
                if not any(self._original_retry(record['outcome'], known.get(key, {}).get('outcome', {}))
                           for key in value['retries']):
                    raise ValueError('resolution requires authenticated successful original-project retry')
            elif value['status'] == 'resolved' and not control:
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
            return self._present(record)

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
                                 if _same_observation(row['outcome'], outcome)), None)
                if existing is not None:
                    if outcome.get('job_incarnation') and not existing['outcome'].get('job_incarnation'):
                        warnings.append('legacy feedback lacks job incarnation; preserving its existing identity')
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

    @staticmethod
    def _identity(root, target, language, metadata=None):
        metadata = metadata or _read(root / 'campaign-identity.json', {})
        if (not target and metadata.get('schema') == 'ascend-native-adaptive-identity-v1'
                and metadata.get('language') == 'ascend-c'):
            target = 'a3'
        target = target or metadata.get('target', 'unknown')
        target = {'native-a3': 'a3', 'native-a5': 'a5'}.get(target, target)
        return target, language or metadata.get('language', 'unknown')

    def import_phase1(self, root, *, target=None, language=None):
        """Import only outcomes committed by the adaptive wave publisher."""
        root = Path(root).resolve()
        report = dict(source_root=str(root), imported=0, existing=0, warnings=[])
        state = _read_import(root / 'state.json', report['warnings'], {})
        if state.get('schema') != 'a5-phase1-adaptive-state-v2':
            report['warnings'].append('no supported committed Phase 1 state found')
            return report
        target, language = self._identity(root, target, language)
        for row in state.get('failed_outcomes', []) + state.get('prior_outcomes', []):
            try:
                spec = row['spec']
                pid = row['project_id']
                # The state commits a project once; proposal/pending waves are excluded.
                branches = [root / 'waves' / wave / 'branches' / pid
                            for wave in state.get('completed_waves', [])]
                branch = next((path for path in branches if (path / 'outcome.json').is_file()), None)
                evidence = {}
                if branch:
                    plan = branch.parent.parent / 'plan.json'
                    if plan.is_file():
                        evidence['plan'] = dict(retained_path=str(plan), sha256=sha(plan.read_bytes()))
                    for name in ('outcome.json', 'execution-evidence.bin', 'memory-delta.bin'):
                        path = branch / name
                        if path.is_file():
                            evidence[name] = dict(retained_path=str(path), sha256=sha(path.read_bytes()))
                verified = row.get('verified') is True
                outcome = dict(source_type='phase1', source_root=str(root), attempt_id=pid,
                    completed=True, kind='project', target=target, language=language,
                    original_spec=spec, task=dict(task_id=pid, project_spec=spec), runtime={},
                    category='verified-success' if verified else row.get('evidence', {}).get('classification', 'incorrect'),
                    verified=verified, result=dict(verified=verified, summary=row.get('summary'),
                        classification=row.get('evidence', {}).get('classification')),
                    evidence=evidence, interpretation='Observation, not verified reusable knowledge')
                report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
            except (KeyError, TypeError, OSError, ValueError) as error:
                report['warnings'].append(f'{root}: unreadable committed project: {error}')
        return report

    def import_optimizer(self, root, *, target=None, language=None):
        """Import terminal standalone jobs, retaining frozen optimizer artifacts."""
        root = Path(root).resolve()
        report = dict(source_root=str(root), imported=0, existing=0, warnings=[])
        if not root.is_dir():
            report['warnings'].append('source root is missing')
            return report
        store = JobStore(root)
        target, language = self._identity(root, target, language)
        for manifest in sorted(root.glob('*/manifest.json')):
            try:
                job = store.get(manifest.parent.name)
                if job['status'] not in {'failed', 'stopped', 'budget-exhausted', 'completed'}:
                    continue
                events = store.events(job['job_id'])
                # Phase 2 owns these observations and their established identities.
                if any(event['kind'].startswith('phase2-') for event in events):
                    continue
                state = next((event['payload'] for event in reversed(events)
                              if event['kind'] == 'optimizer-state'), {})
                best = state.get('best', {})
                result = dict(job_id=job['job_id'], verified=bool(best),
                    classification=job['status'], speedup=best.get('speedup'),
                    result=best.get('content', {}))
                evidence = {}
                for role, ref in dict(baseline=state.get('baseline'), candidate=best.get('source'),
                                      evaluation=best.get('result')).items():
                    if ref:
                        store.read_artifact(job['job_id'], ref['sha256'])
                        evidence[role] = dict(ref, job_root=str(root), job_id=job['job_id'])
                spec = store.submission(job['job_id'])
                observed_target, observed_language = self._identity(root,
                    None if target == 'unknown' else target,
                    None if language == 'unknown' else language,
                    spec.get('memory_snapshot', {}).get('source_identity'))
                # Memory and budgets vary across retries; acceptance inputs do not.
                original = {key: value for key, value in spec.items()
                            if key not in {'memory_snapshot', 'max_attempts', 'max_seconds', 'name'}}
                task = dict(task_id=job['name'] or job['job_id'], submission=original)
                outcome = dict(source_type='optimizer', source_root=str(root),
                    attempt_id=job['job_id'], completed=True, kind='optimizer',
                    target=observed_target, language=observed_language, original_spec=original,
                    task=task, runtime={}, job_incarnation=events[0]['sha256'],
                    verified=bool(best), result=result, evidence=evidence,
                    category=_category(task, result),
                    interpretation='Observation, not verified reusable knowledge')
                report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
            except (KeyError, TypeError, ValueError, OSError) as error:
                report['warnings'].append(f'{manifest}: unreadable optimizer job: {error}')
        return report

    @staticmethod
    def _original_retry(original, retry):
        if (not retry.get('verified') or original.get('source_type') != retry.get('source_type')
                or original.get('original_spec') != retry.get('original_spec')
                or original['target'] not in {'a3', 'a5'}
                or original['target'] != retry.get('target')
                or original['language'] != retry.get('language')
                or original['language'] not in {'ascend-c', 'catlass-dsl'}):
            return False
        try:
            if original['source_type'] == 'phase2' or retry['evidence'].get('result_artifact'):
                baseline = _baseline_identity(original)
                return (baseline is not None and baseline == _baseline_identity(retry)
                        and _verified_retry(retry)
                        and _category(original['task'], retry['result']) == 'verified-success')
            if original['source_type'] == 'optimizer':
                from optimizer.loop import score
                def artifact(outcome, role):
                    ref = outcome['evidence'][role]
                    return json.loads(JobStore(ref['job_root']).read_artifact(ref['job_id'], ref['sha256']))
                baseline = artifact(original, 'baseline')
                retry_baseline = artifact(retry, 'baseline')
                if _baseline_identity(original) is None or _baseline_identity(original) != _baseline_identity(retry):
                    return False
                measured = score(retry_baseline, artifact(retry, 'evaluation'))
                return measured is not None and measured >= 1.0
            plan_ref = retry['evidence']['plan']
            plan_data = Path(plan_ref['retained_path']).read_bytes()
            if sha(plan_data) != plan_ref['sha256']:
                return False
            plan = json.loads(plan_data)
            if not any(project['spec'] == retry['original_spec'] for project in plan['wave']['projects']):
                return False
            ref = retry['evidence']['outcome.json']
            data = Path(ref['retained_path']).read_bytes()
            if sha(data) != ref['sha256']:
                return False
            outcome = json.loads(data)
            evidence = retry['evidence']['execution-evidence.bin']
            payload = Path(evidence['retained_path']).read_bytes()
            return (outcome.get('verified') is True
                    and outcome.get('wave_fingerprint') == plan['wave_fingerprint']
                    and outcome.get('project_id') == retry['original_spec']['project_id']
                    and sha(payload) == evidence['sha256']
                    and outcome.get('evidence_sha256') == evidence['sha256'])
        except (KeyError, ValueError, OSError, TypeError):
            return False

    def link_original_retry(self, feedback_id, retry_id):
        """Record an attempt; only authenticated original acceptance marks it fixed."""
        known = {row['feedback_id']: row for row in self.records()}
        original = known[feedback_id]
        if retry_id not in known:
            raise KeyError(retry_id)
        if feedback_id == retry_id:
            raise ValueError('retry must have a distinct execution identity')
        def merge(current):
            updates = dict(retries=list(dict.fromkeys([*current['retries'], retry_id])))
            # A later unsuccessful experiment does not erase a demonstrated fix.
            # Reload under the publication lock to include concurrently linked attempts.
            attempts = {row['feedback_id']: row['outcome'] for row in self.records()}
            updates['status'] = ('resolved' if any(self._original_retry(
                original['outcome'], attempts.get(key, {})) for key in updates['retries']) else 'open')
            return updates
        return self.update_investigation(feedback_id, merge=merge)

    def import_campaign(self, root):
        root = Path(root).resolve()
        report = dict(source_root=str(root), imported=0, existing=0, warnings=[])
        if not root.is_dir():
            report['warnings'].append('source root is missing')
            return report
        paths = sorted(root.rglob('state.json')) + sorted(root.rglob('qualification.json'))
        supported = 0
        for path in paths:
            state = _read_import(path, report['warnings'])
            if state is None:
                continue
            qualification = path.name == 'qualification.json'
            if qualification and (path.parent / 'state.json').exists() and not state.get('active'):
                continue
            if state.get('phase') != 'optimizer-phase2':
                continue
            supported += 1
            campaign = path.parent
            metadata = _read_import(campaign / 'campaign-identity.json', report['warnings'], {})
            seed = _read_import(campaign / 'seed-memory.json', report['warnings'], {})
            metadata = {**seed.get('source_identity', {}), **metadata,
                        **_read_import(campaign / 'campaign-run.json', report['warnings'], {})}
            rows = {}
            for row in state.get('attempts', []):
                try:
                    rows[row['attempt_id']] = row
                except (KeyError, TypeError) as error:
                    report['warnings'].append(f'{path}: unreadable attempt row: {error}')
            for retained in ([] if qualification else sorted((campaign / 'attempts').glob('*.json'))):
                row = _read_import(retained, report['warnings'])
                if row is None:
                    continue
                try:
                    rows.setdefault(row['attempt_id'], row)
                except (KeyError, TypeError) as error:
                    report['warnings'].append(f'{retained}: unreadable attempt record: {error}')
            for row in rows.values():
                try:
                    result = row['result']
                    outcome = self._outcome(campaign, row, metadata, report)
                    outcome.update(category=_category(row['task'], result), completed=True)
                    report['imported' if self._put(outcome, report['warnings']) else 'existing'] += 1
                except (KeyError, TypeError, ValueError, OSError) as error:
                    report['warnings'].append(f'{campaign}: unreadable attempt record: {error}')
            pending = state.get('pending') if qualification else _read_import(campaign / 'pending.json', report['warnings'])
            if pending:
                try:
                    if pending['attempt_id'] in rows:
                        continue
                    relative = 'tasks/' + pending['task']['task_id'] + '-' + sha(canonical(pending['task']))[:12]
                except (KeyError, TypeError, ValueError) as error:
                    report['warnings'].append(f'{campaign}: unreadable pending record: {error}')
                    continue
                jobs = campaign / relative / 'jobs'
                manifests = sorted(jobs.glob('*/manifest.json'))
                # A pending selection is not evidence of an interrupted execution.
                for manifest in manifests:
                    try:
                        store = JobStore(jobs)
                        job = store.get(manifest.parent.name)
                        events = store.events(job['job_id'])
                        completed = next((event['payload'] for event in reversed(events)
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
                    boundary = None
                    if pending.get('kind') not in {'qualification', 'practice-qualification'}:
                        boundary = next((event['sequence'] for event in reversed(events)
                            if event['kind'] in {'phase2-attempt-request', 'phase2-prepare-request'}
                            and event['payload'].get('attempt_id') == pending['attempt_id']), None)
                        terminal = next((event['sequence'] for event in reversed(events)
                            if event['payload'].get('manifest', {}).get('status') == job['status']), -1)
                        if boundary is None or terminal <= boundary:
                            report['warnings'].append(f'{campaign}:{pending["attempt_id"]}: '
                                'no matching execution boundary followed by terminal evidence; pending selection not imported')
                            continue  # A new pending selection does not own a previous attempt's stop.
                    prepared = next((event['payload'] for event in reversed(events)
                                     if event['kind'] == 'optimizer-state'), {})
                    evaluated = [item for item in prepared.get('history', [])
                                 if 'result' in item and 'candidate_sha256' in item]
                    consumed = sum(event['payload'].get('result') is not None for event in events
                                   if event['kind'] == 'phase2-attempt-result')
                    candidate = prepared.get('pending', {})
                    if (len(evaluated) > consumed or
                            (candidate.get('evaluation') and not candidate.get('history_recorded'))):
                        report['warnings'].append(f'{campaign}:{pending["attempt_id"]}: '
                            'durable evaluated evidence awaits normal attempt-result recovery; import deferred')
                        continue
                    references = {role: prepared[role] for role in ('baseline',) if role in prepared}
                    source = prepared.get('candidate')
                    if source:
                        source_boundary = next((event['sequence'] for event in events
                            if event['kind'] == 'phase2-attempt-request'
                            and event['payload'].get('attempt_id') == pending['attempt_id']), boundary)
                        if source_boundary is not None and any(event['kind'] == 'artifact'
                                and event['sequence'] > source_boundary and event['payload'] == source for event in events):
                            references['candidate'] = source
                        else:
                            report['warnings'].append(f'{campaign}:{pending["attempt_id"]}: '
                                'candidate ownership is not established for this attempt; reference omitted')
                    row = {**pending, 'execution_directory': relative,
                           'result': dict(job_id=job['job_id'], verified=False,
                               classification=job['status'], handles=job.get('handles', []),
                               **references)}
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
        incarnation = None
        job_id = result.get('job_id')
        if job_id and (directory / 'jobs' / job_id / 'events.jsonl').is_file():
            try:
                events = JobStore(directory / 'jobs').events(job_id)
                incarnation = events[0]['sha256'] if events else None
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
            kind=row.get('kind', 'target'), source_type='phase2', original_spec=row['task'], target=target,
            language=metadata.get('language', runtime.get('language', 'unknown')),
            runtime=runtime, job_incarnation=incarnation, task=row['task'], verified=result.get('verified') is True,
            result=result, evidence=evidence, interpretation='Observation, not verified reusable knowledge')
