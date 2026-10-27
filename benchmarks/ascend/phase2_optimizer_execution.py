"""One frozen optimizer task, with Phase 2-owned semantic attempt boundaries.

The same retained optimizer job is used for every retry. RemoteBackend owns
native execution and exact-handle observation; this module never resubmits an
ambiguous operation or interprets model text as a correctness verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from optimizer.agent import OptimizerAgent
from optimizer.cases import validate_submission
from optimizer.control import WorkerStopped
from optimizer.jobs import JobStore, canonical, sha
from optimizer.loop import OptimizationLoop
from optimizer.memory import verify_snapshot
from optimizer.provider import BoundedAgent
from optimizer.transport import exclusive
from optimizer.worker import RemoteBackend, LocalBackend, knowledge_query
from .execution import apply_worker_config, execution_mode, local_identity, preflight_local


class OptimizerTaskExecutor:
    def __init__(self, directory, task, memory_snapshot, config, *, agent=None, backend=None,
                 expected_baseline=None):
        self.directory = Path(directory)
        config = apply_worker_config(config)
        self.task, self.config = task, config
        self.expected_baseline = expected_baseline
        verify_snapshot(memory_snapshot)
        spec = validate_submission(task['submission'])
        self.store = JobStore(self.directory / 'jobs')
        job = self.store.create(spec['source'], spec['entrypoint'], {'signatures': spec['signatures']},
            memory_snapshot, name=task['task_id'], init_args=spec.get('init_args'),
            init_kwargs=spec.get('init_kwargs'),
            idempotency_key='phase2-task:' + str(self.directory.resolve()),
            max_seconds=config.get('phase2_attempt_seconds', 7200))
        self.job_id = job['job_id']
        self.job_directory = self.store.root / self.job_id
        from .actor_control import ActorControl
        incoming = self.directory / 'actor/actor-control.json'
        if incoming.exists():
            control = ActorControl(self.job_directory)
            for message in ActorControl(incoming.parent).status()['messages']:
                control.send(message['key'], message['text'])
        identity = {'task': task, 'runtime': {key: config.get(key) for key in (
            'target', 'profile', 'runtime', 'arch', 'physical_device', 'remote_workspace',
            'cpl_remote', 'validation_wrapper', 'max_cases', 'knowledge', 'model_profile')}}
        if execution_mode(config) == 'local':
            identity['runtime'].update(local_identity(config))
        if config.get('guide'):
            identity['guide_sha256'] = sha(Path(config['guide']).read_bytes())
        previous = self._event('phase2-task')
        if previous is not None and previous != identity:
            raise ValueError('frozen task or execution configuration changed')
        if previous is None:
            self.store.add_event(self.job_id, 'phase2-task', identity)
        self._agent, self._backend = agent, backend
        self.deadline = None

    def set_remaining_seconds(self, remaining_seconds):
        if remaining_seconds <= 0:
            raise WorkerStopped('Phase 2 elapsed-time budget exhausted')
        self.deadline = time.monotonic() + remaining_seconds

    def _event(self, kind):
        return next((event['payload'] for event in reversed(self.store.events(self.job_id))
                     if event['kind'] == kind), None)

    def _state(self):
        return self._event('optimizer-state') or {}

    def _read(self, artifact):
        return json.loads(self.store.read_artifact(self.job_id, artifact['sha256']))

    def _input_reference(self, name, value):
        if value is None:
            return None
        data = canonical(value)
        return {'name': name, 'sha256': sha(data), 'size': len(data)}

    def _loop(self):
        config = self.config
        if execution_mode(config) == 'local' and self._backend is None:
            preflight_local(config)
        backend = self._backend or (LocalBackend(self.store, self.job_id, config)
            if execution_mode(config) == 'local' else RemoteBackend(self.store, self.job_id, config))
        if self.deadline is not None:
            backend.deadline = min(getattr(backend, "deadline", self.deadline), self.deadline)
            transport = getattr(backend, "transport", None)
            if transport is not None:
                transport.deadline = min(getattr(transport, "deadline", self.deadline), self.deadline)
        agent = self._agent
        if agent is None:
            if config.get('env_file'):
                from dotenv import load_dotenv
                load_dotenv(config['env_file'], override=False)
            from benchmarks.a3_experiments import BackendModel
            agent = OptimizerAgent(guide=Path(config['guide']).read_text(),
                model=BackendModel(config.get('model_profile', BackendModel.DEEPSEEK_FLASH.value)))
            remaining = self.store.get(self.job_id)['max_seconds'] - self.store.get(self.job_id)['elapsed_seconds']
            agent = BoundedAgent(agent, deadline=getattr(backend, 'deadline', time.monotonic() + remaining),
                                 stopped=lambda: self.store.get(self.job_id)['stop_requested'])
        return OptimizationLoop(self.store, self.job_id, agent, backend,
            knowledge=knowledge_query(config, self.job_directory), max_cases=config.get('max_cases', 24))

    def _device_lock(self):
        directory = self.config.get('lock_directory')
        base = Path(directory) if directory else Path(self.config['validation_wrapper']).parent / '.optimizer-locks'
        return base / f'{self.config["target"]}-{self.config["physical_device"]}.lock'

    def prepare(self):
        """Execute and freeze the reference once, retaining baseline measurements."""
        with exclusive(self.job_directory / 'worker.lock'), exclusive(self._device_lock(), wait=True,
                stopped=lambda: self.store.get(self.job_id)['stop_requested']):
            if 'baseline' not in self._state():
                loop = self._loop()
                if self.expected_baseline and 'spec' not in loop.state:
                    spec = validate_submission(self.expected_baseline['frozen_submission'])
                    loop.state['spec'] = loop._artifact('frozen-reference-submission', spec)
                    loop._save()
                loop.run(prepare_only=True)
            state = self._state()
            if 'baseline' not in state:
                raise WorkerStopped('reference preparation stopped before a frozen baseline')
            baseline = self._read(state['baseline'])
            expected = self.expected_baseline
            if expected and (baseline['suite_sha256'] != expected['suite_sha256'] or
                    baseline.get('manifest', {}).get('policy') != expected['policy']):
                raise ValueError('retry frozen suite or comparison policy differs from original')
            return baseline

    def attempt(self, attempt_id, *, memory_snapshot=None, feedback=None):
        """Return one evaluated candidate, or an explicit no-candidate failure.

        Retry an interrupted call with the SAME attempt_id. Only start a new id
        after committing its result in the Phase 2 coordinator. Compile actions
        do not reserve evaluated revisions; infrastructure exceptions propagate.
        """
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError('attempt_id must be a nonempty string')
        if memory_snapshot is not None:
            verify_snapshot(memory_snapshot)
        with exclusive(self.job_directory / 'worker.lock'), exclusive(self._device_lock(), wait=True,
                stopped=lambda: self.store.get(self.job_id)['stop_requested']):
            events = self.store.events(self.job_id)
            results = [event['payload'] for event in events if event['kind'] == 'phase2-attempt-result']
            request = {'attempt_id': attempt_id,
                'memory_snapshot': self._input_reference('phase2-attempt-memory', memory_snapshot),
                'feedback': self._input_reference('phase2-attempt-feedback', feedback)}
            recorded = next((event['payload'] for event in events if event['kind'] == 'phase2-attempt-request'
                             and event['payload']['attempt_id'] == attempt_id), None)
            if recorded is not None and recorded != request:
                raise ValueError('resume the pending attempt with its original memory and feedback')
            completed = next((row for row in results if row['attempt_id'] == attempt_id), None)
            if completed is not None:
                return completed
            active = self._event('phase2-attempt-request')
            if active is not None and active['attempt_id'] not in {row['attempt_id'] for row in results}:
                if active != request:
                    raise ValueError('resume the pending attempt with its original memory and feedback')
                row = self.store.get(self.job_id)
                if row['status'] == 'failed' and 'pending' in self._state():
                    row = self.store.recover(self.job_id)
                if row['status'] in {'failed', 'stopped', 'budget-exhausted'} and 'pending' not in self._state():
                    evaluated = [item for item in self._state().get('history', []) if 'result' in item and 'candidate_sha256' in item]
                    if len(evaluated) <= sum(item.get('result') is not None for item in results):
                        self.store.resume(self.job_id, additional_attempts=1, additional_seconds=self.config.get('phase2_attempt_seconds', 7200))
            else:
                row = self.store.get(self.job_id)
                if row['status'] in {'failed', 'stopped', 'budget-exhausted'}:
                    self.store.resume(self.job_id, additional_attempts=1, additional_seconds=self.config.get('phase2_attempt_seconds', 7200))
                for record, value in ((request['memory_snapshot'], memory_snapshot),
                                      (request['feedback'], feedback)):
                    if record is not None:
                        self.store.put_artifact(self.job_id, record['name'], canonical(value))
                self.store.add_event(self.job_id, 'phase2-attempt-request', request)
            loop = self._loop()
            # A crash after evaluation but before this result checkpoint is
            # recovered from optimizer history without another model/native call.
            evaluated = [row for row in loop.state.get('history', []) if 'candidate_sha256' in row and 'result' in row]
            evaluations_completed = sum(row.get('result') is not None for row in results)
            if len(evaluated) <= evaluations_completed or 'pending' in loop.state:
                loop.run(single_evaluation=True, memory_snapshot=memory_snapshot,
                         context={'phase2': {'single_evaluation': True, 'feedback': feedback}})
                evaluated = [row for row in loop.state.get('history', []) if 'candidate_sha256' in row and 'result' in row]
            state = loop.state
            candidate = evaluated[-1] if len(evaluated) > evaluations_completed else None
            if candidate is None and self.store.get(self.job_id)['status'] in {'stopped', 'budget-exhausted', 'failed'}:
                raise WorkerStopped('attempt stopped without a candidate result')
            evidence = candidate.get('content') if candidate else None
            baseline = self._read(state['baseline'])
            verified = bool(evidence and evidence.get('status') == 'verified'
                and evidence.get('candidate_sha256') == candidate['candidate_sha256']
                and evidence.get('suite_sha256') == baseline['suite_sha256']
                and len(evidence.get('cases', [])) == len(baseline['cases'])
                and {row['case'] for row in evidence.get('cases', [])} == {row['case'] for row in baseline['cases']})
            result = {'attempt_id': attempt_id, 'task_id': self.task['task_id'], 'job_id': self.job_id,
                'verified': verified, 'classification': 'verified' if verified else 'actor-unverified',
                'summary': candidate['summary'] if candidate else 'Actor finished without an evaluated candidate',
                'source_sha256': candidate['candidate_sha256'] if candidate else None,
                'candidate': state.get('candidate') if candidate else None,
                'result': evidence, 'baseline': state['baseline'],
                'speedup': candidate.get('speedup') if candidate else None,
                'profiling': (state.get('best', {}).get('profiling') if candidate and
                    state.get('best', {}).get('candidate_sha256') == candidate['candidate_sha256'] else None),
                'handles': self.store.get(self.job_id)['handles']}
            result['result_artifact'] = self.store.put_artifact(self.job_id,
                'phase2-attempt-result.json', canonical(result))
            self.store.add_event(self.job_id, 'phase2-attempt-result', result)
            return result
