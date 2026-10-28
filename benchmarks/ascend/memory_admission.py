"""Background usefulness review, bounded rewriting and accepted-only publication."""
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import fcntl
from pathlib import Path
import time
from threading import Event

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from .project_verifier import ProjectReview
from .subject_memory import StaleMemory, digest
from .time_budget import AttemptTimeBudget
from .memory_status import admission_assessment, admission_report


def treatment_snapshot(baseline, proposal, provenance, memory_id):
    """Replace only proposed bases; unrelated accepted memory stays identical."""
    value = json.loads(baseline)
    removed = {row['memory_id'] for row in proposal['base_versions']}
    value['memories'] = [row for row in value['memories'] if row['memory_id'] not in removed]
    if proposal['base_versions']:
        memory_id = proposal['base_versions'][0]['memory_id']
    value['memories'].append(dict(memory_id=memory_id, proposal=proposal, provenance=provenance))
    return canonical(value)


class MemoryAdmission:
    """One verification worker; callers may keep launching waves on old memory.

    Job files and evaluator directories survive restart. Completed judgments are
    cached; invalid work needs explicit external settlement certification before
    a new isolated trial can run.
    """
    def __init__(self, root, memory, verifier, rewrite, *, max_seconds=3600,
                 parent_remaining=None, clock=time.monotonic, retry_ready=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease = (self.root / '.admission.lock').open('a+b')
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise RuntimeError('memory verification worker is already active') from None
        self.memory, self.verifier, self.rewrite = memory, verifier, rewrite
        self.seconds, self.parent_remaining, self.clock = max_seconds, parent_remaining, clock
        # Certification must establish all owned operations terminal AND that
        # a fresh isolated trial is eligible. Absence of a receipt is not proof.
        self.retry_ready = retry_ready
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='memory-verifier')
        self.stop_requested = Event()
        self.future, self.active = None, None
        self.delivered = set()
        # Resume interrupted orchestration, not settled uncertainty by default.
        for path in self.root.glob('*/job.json'):
            job = json.loads(path.read_bytes())
            if self._resumable(job):
                job['status'] = 'pending'
                atomic(path, canonical(job))

    def _resumable(self, job):
        if job['status'] == 'evaluating':
            return True
        if job['status'] != 'inconclusive':
            return False
        return bool(job.get('error_type')) or self.retry_ready is not None

    def _path(self, review_id):
        return self.root / digest(review_id) / 'job.json'

    def _save(self, job):
        path = self._path(job['review_id'])
        path.parent.mkdir(exist_ok=True)
        with self.memory.transaction() as state:
            manual = state.get('manual_overrides', {}).get(job['review_id'])
            if manual is not None:
                job.setdefault('status_before_override', job['status'])
                job.update(status='accepted', manual_override=manual, admission=manual['admission'])
            job['assessment'] = admission_assessment(job)
            atomic(path, canonical(job))

    def submit(self, review_id, proposal, provenance, request, *, remaining_seconds):
        path = self._path(review_id)
        if not path.exists():
            job = dict(review_id=review_id, proposal=proposal, provenance=provenance,
                request=request, baseline=base64.b64encode(self.memory.snapshot()).decode(),
                status='pending', rewrites=0, revision=0, trial_task=None, evaluations=[],
                sequence=len(list(self.root.glob('*/job.json'))),
                trial_ordinal=0, probe_index=0, trial_history_length=0,
                allocation=min(self.seconds, remaining_seconds))
            self._save(job)
        self._start()

    def _start(self):
        if self.stop_requested.is_set():
            return
        if self.future is not None:
            return
        for path in sorted(self.root.glob('*/job.json')):
            job = json.loads(path.read_bytes())
            if job['status'] == 'pending':
                self.active = job['review_id']
                self.future = self.pool.submit(self._work, job)
                return

    def _work(self, job):
        directory = self._path(job['review_id']).parent
        budget = AttemptTimeBudget(directory / 'elapsed.json', job['allocation'], clock=self.clock)
        def remaining():
            if self.stop_requested.is_set():
                raise TimeoutError('memory worker paused; resume retained evaluation')
            seconds = budget.deadline - self.clock()
            if self.parent_remaining is not None:
                seconds = min(seconds, self.parent_remaining())
            if seconds <= 0:
                raise TimeoutError('memory admission elapsed allocation exhausted')
            return seconds
        try:
            while True:
                remaining()
                with self.memory.transaction() as state:
                    forced = job['review_id'] in state.get('manual_overrides', {})
                if forced:
                    self._save(job)
                    return
                job['status'] = 'evaluating'
                job.pop('error_type', None)
                job.pop('error', None)
                self._save(job)
                if job.get('manual_override'):
                    return
                # Reconcile a concurrent accepted replacement before testing this draft.
                if not self.memory.bases_current(job['proposal']):
                    rewritten = self._rewrite(job, dict(reason='Accepted base version changed',
                        instructions='Reconcile against current accepted memories; preserve useful prior content.',
                        current_memories=self.memory.records()), remaining())
                    if not self._apply_rewrite(job, rewritten):
                        return
                    job['baseline'] = base64.b64encode(self.memory.snapshot()).decode()
                    job['trial_task'] = None
                    self._save(job)
                baseline = base64.b64decode(job['baseline'], validate=True)
                treatment = treatment_snapshot(baseline, job['proposal'], job['provenance'], job['review_id'])
                trial_directory = directory / f"revision-{job['revision']}"
                if job.get('probe_index', 0):
                    trial_directory /= f"probe-{job['probe_index']}"
                from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
                shared = ContentAddressedArtifacts(self.memory.root.parent / 'artifacts')
                history = job['evaluations'][:job.get('trial_history_length', 0)]
                options = dict(check_remaining=remaining)
                if getattr(self.verifier, 'mode', None) == 'focused':
                    options['source_request'] = job['request']
                if history or job.get('probe_request') is not None:
                    options.update(history=history, probe_request=job.get('probe_request'),
                        prior_task=job.get('prior_task'), read_history_artifact=shared.read_bytes)
                result = self.verifier.evaluate(job['proposal'], baseline, treatment,
                    trial_directory, remaining_seconds=remaining(), task=job['trial_task'],
                    sequence=job.get('sequence',0)+job.get('trial_ordinal', job['revision']), **options)
                trial_store = ContentAddressedArtifacts(trial_directory / 'artifacts')
                for sha in result.get('artifacts', {}).values():
                    shared.put_bytes(trial_store.read_bytes(sha))
                job.setdefault('trace_sha256', []).append(shared.put_json(getattr(self.verifier,'last_trace',[])))
                if not job['evaluations'] or digest(job['evaluations'][-1]) != digest(result):
                    job['evaluations'].append(result)
                job['trial_task'] = result.get('trial_task')
                self._save(job)
                if job.get('manual_override'):
                    return
                remaining()
                if result['decision'] == 'inconclusive':
                    valid = result.get('validity') == {'control': True, 'treatment': True}
                    probe = result.get('next_probe') if valid else None
                    retry = False
                    if not valid and self.retry_ready is not None:
                        retry = self.retry_ready(result, trial_directory,
                                                 remaining_seconds=remaining()) is True
                    if probe is not None or retry:
                        job.update(prior_task=job['trial_task'], trial_task=None,
                            probe_request=probe, probe_index=job.get('probe_index', 0)+1,
                            trial_ordinal=job.get('trial_ordinal', job['revision'])+1,
                            trial_history_length=len(job['evaluations']))
                        if retry:
                            # Retry the same frozen task in NEW isolated arms,
                            # only after the external settlement certification.
                            job['trial_task'] = job['prior_task']
                        self._save(job)
                        continue
                    job['status'] = 'inconclusive'
                    self._save(job)
                    return
                if result['decision'] == 'accept':
                    try:
                        job['admission'] = self.memory.accept(job['review_id'], job['proposal'], dict(
                            job['provenance'], memory_verification=dict(
                                evaluation_id=result.get('evaluation_id',digest(result)),
                                decision=result['decision'], evidence=result['evidence'],
                                **({key: result[key] for key in ('mode', 'claim_kind', 'evidence_stages')}
                                   if result.get('mode') == 'focused' else {}),
                                trace_sha256=job['trace_sha256'][-1], rewrites=job['rewrites'],
                                evaluations=[row.get('evaluation_id', digest(row)) for row in job['evaluations']],
                                traces=job['trace_sha256'])),
                            check_remaining=remaining)
                    except StaleMemory:
                        continue
                    job['status'] = 'accepted'
                    self._save(job)
                    return
                # Both reject and rewrite first receive the requested two rewrite opportunities.
                if job['rewrites'] == 2:
                    job['status'] = 'rejected'
                    self._save(job)
                    return
                rewritten = self._rewrite(job, result, remaining())
                if not self._apply_rewrite(job, rewritten):
                    return
                job['rewrites'] += 1
                self._save(job)
        except (TimeoutError, OSError, InfrastructureFailure) as error:
            job.update(status='inconclusive', error_type=type(error).__name__, error=str(error))
            self._save(job)
        finally:
            budget.close()

    def _rewrite(self, job, feedback, seconds):
        request = dict(job['request'])
        refs = dict(request['artifacts'])
        for name, arm in feedback.get('arms', {}).items():
            if arm is not None:
                refs.update({f'memory-trial:{name}:{role}': sha for role, sha in arm['artifacts'].items()})
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        shared = ContentAddressedArtifacts(self.memory.root.parent / 'artifacts')
        proposal_sha = shared.put_bytes(canonical(job['proposal']))
        feedback_sha = shared.put_bytes(canonical(feedback))
        refs.update({'memory-rewrite:previous-proposal': proposal_sha,
                     'memory-rewrite:usefulness-feedback': feedback_sha})
        request['artifacts'] = refs
        previous = {key: str(job['proposal'].get(key, ''))[:512] for key in
                    ('summary', 'applicability', 'preconditions', 'expectations', 'limitations')}
        previous['sha256'] = proposal_sha
        usefulness = {key: str(feedback.get(key, ''))[:2048] for key in
                      ('decision', 'reason', 'instructions', 'evaluation_id')}
        usefulness.update(sha256=feedback_sha, evidence=feedback.get('evidence', [])[:16],
                          arms={name: dict(host_verified=arm.get('host_verified'),
                                          source_sha256=arm.get('source_sha256'),
                                          artifact_count=len(arm.get('artifacts', {})))
                                for name, arm in feedback.get('arms', {}).items() if arm is not None})
        result = self.rewrite(ProjectReview(**request), feedback=dict(
            previous_proposal=previous, usefulness=usefulness,
            note='Full immutable previous proposal and usefulness feedback are admitted artifacts; read pages by SHA for exact examples, trial evidence and detailed judgments.'), max_seconds=seconds)
        return result

    def _apply_rewrite(self, job, review):
        job.setdefault('rewrite_reviews', []).append(review)
        if review['decision'] == 'repair':
            job.update(status='rejected', project_repair=review)
            self._save(job)
            return False
        if review['memory'] is None:
            job.update(status='rejected')
            self._save(job)
            return False
        job['proposal'] = review['memory']
        # A rewrite may select a different currently accepted merge target.
        job['baseline'] = base64.b64encode(self.memory.snapshot()).decode()
        job['revision'] += 1
        job.update(probe_index=0, probe_request=None, prior_task=None,
            trial_ordinal=job.get('trial_ordinal', job['revision']-1)+1,
            trial_history_length=len(job['evaluations']))
        self._save(job)
        return True

    def poll(self, *, remaining_seconds):
        if self.future is not None and self.future.done():
            self.future.result()
            self.future, self.active = None, None
        result = []
        for path in sorted(self.root.glob('*/job.json')):
            job = json.loads(path.read_bytes())
            with self.memory.transaction() as state:
                manual = state.get('manual_overrides', {}).get(job['review_id'])
            delivery = ('manual:' if manual else '') + job['review_id']
            if manual is not None:
                job.update(status='accepted', manual_override=manual)
            if job['status'] in {'accepted', 'rejected'} and delivery not in self.delivered:
                judgment = job['evaluations'][-1] if job['evaluations'] else {}
                result.append(dict(review_id=job['review_id'], delivery_id=delivery, manual_override=manual, outcome=job['status'],
                    assessment=admission_assessment(job),
                    judgment={key:judgment.get(key) for key in ('decision','reason','evidence','instructions','evaluation_id')},
                    rewrites=job['rewrites'], repair=None if manual else job.get('project_repair'), provenance=job['provenance']))
        self._start()
        return result

    def report(self):
        """Expose held reviews without resolving scheduler work or restarting it."""
        return admission_report(self.root, memory_root=self.memory.root)

    def acknowledge(self, review_id):
        self.delivered.add(review_id)

    def wait(self, *, remaining_seconds):
        self._start()
        if self.future is None:
            raise TimeoutError('memory review remains inconclusive; retained work needs resuming')
        try:
            self.future.result(timeout=remaining_seconds)
        except FutureTimeout:
            raise TimeoutError('waiting for memory review exceeded campaign allocation') from None

    def start_session(self):
        if self.lease.closed:
            self.lease = (self.root / '.admission.lock').open('a+b')
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lease.close()
                raise RuntimeError('memory verification worker is already active') from None
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='memory-verifier')
            self.future, self.active = None, None
        self.stop_requested.clear()
        for path in self.root.glob('*/job.json'):
            job=json.loads(path.read_bytes())
            if self._resumable(job):
                job['status']='pending'
                self._save(job)

    def close(self):
        if not self.lease.closed:
            self.stop_requested.set()
            self.pool.shutdown(wait=True)
            self.lease.close()
