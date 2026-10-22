"""Opt-in failure obligations, separate from the campaign's project target."""
import base64
from contextlib import ExitStack, contextmanager
import math
import threading
import time
import json
from copy import copy
from pathlib import Path

from optimizer.jobs import atomic, canonical, sha
from .failure_curriculum import FailureCurriculum
from .phase2_feedback import FeedbackStore


class FailureTimeBudget:
    """Start the shared upstream attempt clock only for attributable work."""
    def __init__(self, path, seconds, *, clock=time.monotonic):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('failure seconds must be positive and finite')
        self.path, self.seconds, self.clock = path, seconds, clock
        self.timer, self.depth, self.lock = None, 0, threading.RLock()
        if path.exists() and json.loads(path.read_bytes())['max_seconds'] != seconds:
            raise ValueError('failure time allowance changed')

    @property
    def elapsed_seconds(self):
        with self.lock:
            if self.timer:
                return self.timer.elapsed + max(0, self.clock() - self.timer.started)
            return json.loads(self.path.read_bytes())['elapsed_seconds'] if self.path.exists() else 0

    @property
    def remaining_seconds(self):
        return max(0, self.seconds - self.elapsed_seconds)

    @property
    def expired(self):
        return self.remaining_seconds <= 0

    @contextmanager
    def active(self):
        from .time_budget import AttemptTimeBudget
        with self.lock:
            if self.timer is None:
                self.timer = AttemptTimeBudget(self.path, self.seconds, clock=self.clock)
            self.depth += 1
        try:
            yield
        finally:
            with self.lock:
                self.depth -= 1
                if self.depth == 0:
                    self.timer.close()
                    self.timer = None


class AutomaticRemediation:
    def __init__(self, coordinator, *, target, store=None, sources=(), seconds=1800,
                 retry_external=None):
        self.runner, self.target = coordinator, target
        self.store = FeedbackStore(store or coordinator.root / 'feedback')
        self.sources, self.seconds = sources, seconds
        self.retry_external = retry_external
        self.path = coordinator.root / 'remediation.json'
        self.state = json.loads(self.path.read_bytes()) if self.path.exists() else {
            'schema': 'automatic-remediation-v1', 'failures': {}, 'retry_feedback_ids': []}
        self.feedback = FailureCurriculum(coordinator.root, self.store.root,
                                          dict(selected=[], controls=[]))
        coordinator.failure_feedback = self.feedback
        coordinator.curriculum.failure_feedback = self.feedback
        coordinator.automatic_remediation = self

    def wave_slots(self, state):
        outcomes = self.runner._seed_context_state(state)["prior_outcomes"]
        return max(self.runner.max_parallel, 1 if any(row["verified"] for row in outcomes) else 2)

    def save(self):
        atomic(self.path, canonical(self.state))

    def refresh(self, state):
        self.store.import_phase1(self.runner.root, target=self.target,
            language='ascend-c' if self.target == 'a3' else 'catlass-dsl')
        for source in self.sources:
            kind, root = source['type'], source['root']
            {'phase1': self.store.import_phase1, 'phase2': self.store.import_campaign,
             'optimizer': self.store.import_optimizer}[kind](root)
        self.feedback.refresh(self.target)
        retry_parent = (self.runner.root / 'original-retries').resolve()
        excluded = set(self.state['retry_feedback_ids']) | {
            row['feedback_id'] for row in self.store.records()
            if Path(row['outcome']['source_root']).resolve().is_relative_to(retry_parent)}
        self.feedback.selection['selected'] = [row for row in self.feedback.selection['selected']
                                              if row['feedback_id'] not in excluded]
        self.budgets = {}
        for row in self.feedback.selection['selected']:
            key = row['feedback_id']
            budget = FailureTimeBudget(self.runner.root / 'failure-budgets' / (key + '.json'), self.seconds, clock=self.runner.clock)
            self.budgets[key] = budget
            self.state['failures'].setdefault(key, dict(retries=[], pending=None))
            self.state['failures'][key].update(active_seconds=budget.elapsed_seconds,
                                              remaining_seconds=budget.remaining_seconds)
        self.feedback.selection['selected'] = [row for row in self.feedback.selection['selected']
                                               if not self.budgets[row['feedback_id']].expired]
        self.feedback.ids = {row['feedback_id'] for row in self.feedback.selection['selected']}
        self.save()
        for key, failure in self.state['failures'].items():
            for retry in failure['retries']:
                directory = self.runner.root / 'original-retries' / key / Path(retry['root']).name
                self._assess_retry(directory)
        return bool(self.feedback.ids)

    @contextmanager
    def active(self, keys=None):
        with ExitStack() as stack:
            for key in sorted(self.feedback.ids if keys is None else keys):
                stack.enter_context(self.budgets[key].active())
            yield

    def project_keys(self, request):
        branch = request.branch_root.resolve()
        retry_parent = (self.runner.root / 'original-retries').resolve()
        if branch.is_relative_to(retry_parent):
            return {branch.relative_to(retry_parent).parts[0]}
        rows = list(self.feedback.lineage)
        pending = self.runner.root / 'pending.json'
        if pending.exists():
            rows += json.loads(pending.read_bytes()).get('failure_investigations', [])
        keys = {row['feedback_id'] for row in rows
                if request.assignment.project_id in {row.get('project_id'), row.get('control_project_id')}
                and row['feedback_id'] in self.feedback.ids}
        return keys

    def project_active(self, request):
        return self.active(self.project_keys(request))

    def project_remaining(self, request):
        return min([self.runner._remaining_seconds(),
                    *(self.budgets[key].remaining_seconds for key in self.project_keys(request))])

    def retry_ready(self, state):
        for row in self.feedback.selection['selected']:
            key = row['feedback_id']
            if row['investigation']['status'] != 'ready-to-retry':
                continue
            obligation = self.state['failures'][key]
            if obligation['pending'] is None:
                obligation['pending'] = 'retry-' + str(len(obligation['retries']) + 1)
                self.save()
            directory = self.runner.root / 'original-retries' / key / obligation['pending']
            with self.budgets[key].active():
                if row['outcome']['source_type'] == 'phase1':
                    self._phase1_retry(row, directory, state)
                elif self.retry_external:
                    self.retry_external(row, directory, min(self.budgets[key].remaining_seconds, self.runner._remaining_seconds()))
                else:
                    raise ValueError('external feedback requires a configured optimizer retry adapter')
            attempts = [r for r in self.store.records()
                        if r['outcome']['source_root'] == str(directory.resolve())]
            if not attempts:
                raise ValueError('original retry did not retain a committed observation')
            for attempt in attempts:
                retry_id = attempt['feedback_id']
                self.store.link_original_retry(key, retry_id)
                self.state['retry_feedback_ids'].append(retry_id)
            obligation['retries'].append(dict(root=str(directory),
                feedback_ids=[r['feedback_id'] for r in attempts]))
            obligation['pending'] = None
            original = next(r for r in self.store.records() if r['feedback_id'] == key)
            if original['lifecycle'] == 'failed':
                latest = next((r for r in reversed(self.feedback.lineage) if r['feedback_id'] == key), None)
                if latest:
                    self.feedback.lineage.append(dict(latest, status='investigating',
                        reason='Original acceptance still fails after retry; investigate retained retry evidence.'))
                    self.feedback.record([], state)
            self.save()
            self._assess_retry(directory)

    def _assess_retry(self, directory):
        verifier = getattr(self.runner, 'usefulness_verifier', None)
        marker = directory / 'usefulness-assessment.json'
        if (verifier is None or not verifier.enabled or not (directory / 'retry-memory.bin').exists()
                or marker.exists() and json.loads(marker.read_bytes()).get('complete')):
            return
        try:
            from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
            state = json.loads((directory / 'state.json').read_bytes())
            prior = (directory / 'retry-memory.bin').read_bytes()
            for wave in state['completed_waves']:
                for outcome in (directory / 'waves' / wave / 'branches').glob('*/outcome.json'):
                    if not json.loads(outcome.read_bytes())['verified']:
                        continue
                    delta = json.loads((outcome.parent / 'memory-delta.bin').read_bytes())
                    memory = Phase1ProjectMemory.from_mapping(json.loads(base64.b64decode(delta['memory_base64'])))
                    source_id = outcome.parent.relative_to(self.runner.root).as_posix()
                    verifier.assess_memory(memory, prior, source_id)
            atomic(marker, canonical(dict(complete=True)))
        except Exception as error:
            # Acceptance and linked lifecycle have already committed. Assessment
            # is optional evidence and cannot undo or delay that transition.
            atomic(marker, canonical(dict(complete=False, error_type=type(error).__name__, error=str(error))))

    def _phase1_retry(self, row, directory, state, *, retain_campaign=False):
        from benchmarks.a5kernels.phase1_curriculum import ProjectSpec, WaveProject, WaveSpec
        from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator
        spec = ProjectSpec.from_mapping(row['outcome']['original_spec'])
        snapshot = self.runner._memory_snapshot(state)
        pending_path = directory / 'retry-memory.bin'
        directory.mkdir(parents=True, exist_ok=True)
        if pending_path.exists():
            snapshot = pending_path.read_bytes()
        else:
            pending_path.write_bytes(snapshot)
        wave = WaveSpec('original-retry', sha(snapshot), (), (WaveProject(spec),))
        live = LiveWaveCoordinator(directory / 'waves' / wave.wave_id, wave, snapshot,
                                   self.runner._execute, 1, allow_semantic_failures=True)
        result = live.resume()
        pending = dict(session=state['session'])
        blank = state if retain_campaign else dict(state, prior_outcomes=[], failed_outcomes=[], memory_bytes_base64=[],
                     coverage=[], completed_waves=[], wave_index=0)
        deltas = tuple(self._retry_delta(directory, wave.wave_id, spec.project_id)
                       if result.commits else None for _ in wave.projects)
        publisher = copy(self.runner)
        publisher.root = directory
        published = publisher._publish(blank, wave, pending, deltas)
        # _publish reads failed evidence relative to runner.root; use a child publisher.
        atomic(directory / 'state.json', canonical(published))
        self.store.import_phase1(directory, target=self.target,
            language='ascend-c' if self.target == 'a3' else 'catlass-dsl')

    def _retry_delta(self, root, wave, project):
        return json.loads((root / 'waves' / wave / 'branches' / project / 'memory-delta.bin').read_bytes())


def optimizer_retry_adapter(runner, store, config):
    """Retry external frozen acceptance using current committed Phase 1 knowledge."""
    from .failure_curriculum import retry_spec
    from .phase2_optimizer_execution import OptimizerTaskExecutor
    from optimizer.memory import freeze_committed_phase1

    def retry(row, directory, seconds):
        memory_path = directory / 'seed-memory.json'
        directory.mkdir(parents=True, exist_ok=True)
        if memory_path.exists():
            memory = json.loads(memory_path.read_bytes())
        else:
            memory = freeze_committed_phase1(runner.root, runner.initial_memory)
            atomic(memory_path, canonical(memory))
        expected = retry_spec(dict(selected=[row]), 'a3')
        task = expected['task']
        if row['outcome']['source_type'] == 'optimizer':
            submission = dict(task['submission'])
            submission['signatures'] = submission.pop('signature')['signatures']
            task = dict(task, submission=submission)
        bound = dict(config, phase2_attempt_seconds=max(1, seconds))
        executor = OptimizerTaskExecutor(directory, task, memory, bound, expected_baseline=expected)
        executor.prepare()
        result = executor.attempt('original-retry', memory_snapshot=memory)
        attempt = dict(attempt_id='original-retry', task=row['outcome']['task'], result=result,
                       execution_directory='', kind='target')
        atomic(directory / 'state.json', canonical(dict(phase='optimizer-phase2', attempts=[attempt])))
        atomic(directory / 'campaign-identity.json', canonical(dict(target='a3', language='ascend-c')))
        report = dict(warnings=[])
        outcome = store._outcome(directory, attempt, dict(target='a3', language='ascend-c'), report)
        outcome.update(source_type=row['outcome']['source_type'],
            original_spec=row['outcome']['original_spec'], completed=True)
        from .phase2_feedback import _category
        outcome['category'] = _category(outcome['task'], result)
        store._put(outcome, report['warnings'])
    return retry
