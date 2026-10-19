"""Durable backend-neutral optimizer. Backends own retained-operation recovery.

prepare(spec, directory) freezes reference/inputs and returns suite_sha256 plus
cases [{case, contract, timing:{torch:{latency_us}}}]. evaluate(source,directory,
block_count=) returns native.evaluate JSON (profile=False, benchmark=True).
profile(source,directory,block_count=) returns compact diagnostics. Every backend
call MUST be idempotent for its directory and observe existing handles on retry.
"""
from __future__ import annotations

import copy
import json
import math
import time

from .cases import validate_submission
from .jobs import canonical, sha
from .memory import verify_snapshot
from .control import ObservationUnavailable, WorkerStopped


def score(baseline, result):
    """Equal-weight geomean using pinned Torch timings over the entire suite."""
    if result.get('status') != 'verified' or result.get('suite_sha256') != baseline['suite_sha256']:
        return None
    expected = {row['case']: row for row in baseline['cases']}
    actual = {row['case']: row for row in result['cases']}
    if not expected or set(actual) != set(expected) or len(actual) != len(result['cases']):
        return None
    ratios = []
    for key, row in expected.items():
        a = row.get('timing', {}).get('torch', {}).get('latency_us')
        b = actual[key].get('timing', {}).get('ascendc', {}).get('latency_us')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (a, b)):
            return None
        ratios.append(a / b)
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


class OptimizationLoop:
    def __init__(self, store, job_id, agent, backend, *, knowledge=None, clock=time.time):
        self.store, self.key, self.agent, self.backend = store, job_id, agent, backend
        self.knowledge, self.clock = knowledge, clock
        self.directory = store.root / job_id / 'execution'
        self.directory.mkdir(exist_ok=True)
        self.state = next((copy.deepcopy(e['payload']) for e in reversed(store.events(job_id))
                           if e['kind'] == 'optimizer-state'),
                          {'seen': [], 'history': [], 'turn': 0})

    def _save(self):
        self.store.add_event(self.key, 'optimizer-state', self.state)

    def _artifact(self, name, value):
        return self.store.put_artifact(self.key, name, canonical(value))

    def _read(self, record):
        return json.loads(self.store.read_artifact(self.key, record['sha256']))

    def _record(self, kind, payload):
        artifact = self._artifact('model-' + kind, payload)
        self.store.add_event(self.key, 'model-' + kind, artifact)

    def _boundary(self):
        row = self.store.get(self.key)
        if row['stop_requested']:
            if row['status'] == 'running':
                self.store.transition(self.key, 'stopped')
            return False
        return row['status'] == 'running'

    def _action(self, stage, callback):
        if not self._boundary():
            return None
        self.store.transition(self.key, 'running', stage=stage)
        started = self.clock()
        self.state['inflight_started'] = started
        self._save()
        try:
            value = callback()
            # Persist returned evidence before clearing the in-flight marker.
            self.state['action_result'] = self._artifact(stage, value)
            return value
        finally:
            self.store.account_time(self.key, max(0, self.clock() - started))
            self.state.pop('inflight_started', None)
            self._save()

    def _spec(self, submission):
        signature = submission['signature']
        if isinstance(signature, dict) and 'signatures' in signature:
            if set(signature) != {'signatures'}:
                raise ValueError('signature envelope contains unexpected fields')
            signatures = signature['signatures']
        else:
            # Preserve jobs created by the original CLI's list/single-case API.
            signatures = signature if isinstance(signature, list) else [signature]
        return validate_submission({k: submission[k] for k in ('source', 'entrypoint', 'init_args', 'init_kwargs')}
                                   | {'signatures': signatures})

    def run(self):
        row = self.store.get(self.key)
        if row['status'] not in {'queued', 'running'}:
            return row
        submission = self.store.submission(self.key)
        verify_snapshot(submission['memory_snapshot'])
        identity = self.agent.identity
        if 'model' in self.state and self.state['model'] != identity:
            raise ValueError('optimizer model identity changed')
        self.state['model'] = identity
        if 'inflight_started' in self.state:
            self.store.account_time(self.key, max(0, self.clock() - self.state.pop('inflight_started')))
        self._save()
        if self.store.get(self.key)['status'] == 'queued':
            self.store.transition(self.key, 'running')
        try:
            self._run(submission)
        except ObservationUnavailable:
            if self.store.get(self.key)['status'] == 'running':
                self.store.transition(self.key, 'running', stage='observation-unavailable')
            raise
        except WorkerStopped:
            row = self.store.get(self.key)
            pending = self.state.get('pending')
            if pending is not None and 'evaluation' not in pending:
                self.state['history'].append({'candidate_sha256': pending['sha256'],
                                             'status': 'cancelled', 'summary': pending['summary']})
                self.state.pop('pending')
                self._save()
            if row['status'] == 'running':
                self.store.transition(self.key, 'stopped' if row['stop_requested'] else 'budget-exhausted')
            raise
        except Exception as exc:
            self.store.add_event(self.key, 'optimizer-error', {'type': type(exc).__name__, 'message': str(exc)})
            if self.store.get(self.key)['status'] == 'running':
                self.store.transition(self.key, 'failed')
            raise
        return self.store.get(self.key)

    def _run(self, submission):
        spec = self._spec(submission)
        if 'spec' not in self.state:
            action = self._model('cases', {'submission': spec})
            if action is None:
                return
            if set(action) != {'action', 'signatures', 'summary'} or action['action'] != 'cases':
                raise ValueError('first action must propose additional signatures')
            additions = action['signatures']
            if not isinstance(additions, list) or len(spec['signatures']) + len(additions) > 24:
                raise ValueError('case proposals exceed 24-case budget')
            accepted, rejected = [], []
            for signature in additions:
                try:
                    validate_submission({**spec, 'signatures': [signature]})
                    accepted.append(signature)
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append({'signature': signature, 'error': str(exc)})
            self.store.add_event(self.key, 'case-proposals', {'accepted': accepted, 'rejected': rejected,
                                                           'summary': action['summary']})
            spec['signatures'] += accepted
            self.state['spec'] = self._artifact('frozen-reference-submission', spec)
            self._save()
        spec = self._read(self.state['spec'])
        if 'baseline' not in self.state:
            baseline = self._action('prepare', lambda: self.backend.prepare(spec, self.directory / 'baseline'))
            if baseline is None:
                return
            if not baseline.get('suite_sha256') or not baseline.get('cases'):
                raise ValueError('backend must return frozen nonempty baseline')
            self.state['baseline'] = self._artifact('baseline', baseline)
            self._save()
        baseline = self._read(self.state['baseline'])
        while self._boundary():
            pending = self.state.get('pending')
            if pending is None:
                action = self._model('optimize', {'submission': spec, 'baseline': baseline,
                        'history': self.state['history'], 'best': self.state.get('best'),
                        'memory_summary': [{'index': i, 'summary': str(entry)[:1200]} for i, entry
                            in enumerate(submission['memory_snapshot']['memory_entries'])]})
                if action is None:
                    break
                kind = action.get('action')
                if kind == 'finish' and set(action) == {'action', 'summary'}:
                    self.store.add_event(self.key, 'finished', action)
                    self.store.transition(self.key, 'completed')
                    break
                if kind in {'query-knowledge', 'read-memory'}:
                    self._tool(action, submission['memory_snapshot'])
                    continue
                if (set(action) != {'action', 'source', 'block_count', 'summary'} or kind != 'candidate'
                        or not isinstance(action['source'], str) or not action['source'].strip()
                        or type(action['block_count']) is not int or not 1 <= action['block_count'] <= 65535):
                    raise ValueError('invalid candidate action')
                source_sha = sha(action['source'].encode())
                identity = sha(canonical({'source_sha256': source_sha, 'block_count': action['block_count']}))
                if identity in self.state['seen']:
                    self.state['history'].append({'candidate_sha256': source_sha,
                        'configuration_sha256': identity, 'block_count': action['block_count'], 'status': 'duplicate-skipped'})
                    self._save()
                    continue
                pending = {**action, 'sha256': source_sha, 'configuration_sha256': identity,
                           'attempt': self.store.get(self.key)['attempts'] + 1}
                self.state['pending'] = pending
                self.state['candidate'] = self._artifact('candidate.cpp', action['source'])
                self._save()
            if not self._boundary():
                break
            # Attempt is reserved before submission: crashes never lose its charge.
            if self.store.get(self.key)['attempts'] < pending['attempt']:
                self.store.record_attempt(self.key, candidate_sha256=pending['sha256'], result='evaluating')
            # Final allowed attempt must still run after its reservation exhausts count.
            if 'evaluation' not in pending:
                result = self._evaluate_reserved(pending)
                pending['evaluation'] = self._artifact('evaluation', result)
                self._save()
            result = self._read(pending['evaluation'])
            value = score(baseline, result)
            history = {'candidate_sha256': pending['sha256'], 'result': pending['evaluation'],
                       'configuration_sha256': pending.get('configuration_sha256', pending['sha256']),
                       'block_count': pending['block_count'],
                       'content': result, 'speedup': value, 'summary': pending['summary']}
            if not pending.get('history_recorded'):
                self.state['history'].append(history)
                self.state['seen'].append(pending.get('configuration_sha256', pending['sha256']))
                pending['history_recorded'] = True
            if value is not None and (pending.get('new_best') or value > self.state.get('best', {}).get('speedup', 0)):
                pending['new_best'] = True
                self.state['best'] = {**history, 'source': self.state['candidate'],
                                     'block_count': pending['block_count'], 'faster_than_torch': value > 1}
                row = self.store.get(self.key)
                if not row['stop_requested'] and row['elapsed_seconds'] < row['max_seconds']:
                    try:
                        self.store.add_event(self.key, 'reserved-stage', {'stage': 'profile'})
                        report = self._reserved_action(lambda: self.backend.profile(pending['source'],
                            self.directory / pending.get('configuration_sha256', pending['sha256']) / 'profile', block_count=pending['block_count']))
                        self.state['best']['profiling'] = self._artifact('profiling', report)
                    except (ObservationUnavailable, WorkerStopped):
                        raise
                    except Exception as exc:
                        self.store.add_event(self.key, 'profiling-unavailable', {'error': str(exc)})
                else:
                    self.store.add_event(self.key, 'profiling-unavailable',
                                         {'reason': 'budget or stop boundary', 'candidate_sha256': pending['sha256']})
            self.state.pop('pending')
            self._save()

    def _evaluate_reserved(self, pending):
        return self._reserved_action(lambda: self.backend.evaluate(pending['source'],
            self.directory / pending.get('configuration_sha256', pending['sha256']), block_count=pending['block_count']))

    def _reserved_action(self, callback):
        started = self.clock()
        self.state['inflight_started'] = started
        self._save()
        try:
            return callback()
        finally:
            self.store.account_time(self.key, max(0, self.clock() - started))
            self.state.pop('inflight_started', None)
            self._save()

    def _model(self, stage, context):
        if self.state['turn'] >= 200:
            raise ValueError('model action limit reached')
        self.state['turn'] += 1
        self._save()
        return self._action(stage, lambda: self.agent.complete(stage, context, self._record))

    def _tool(self, action, snapshot):
        if action['action'] == 'read-memory' and set(action) == {'action', 'index'}:
            index = action['index']
            if type(index) is not int or not 0 <= index < len(snapshot['memory_entries']):
                raise ValueError('invalid memory index')
            result = snapshot['memory_entries'][index]
        elif action['action'] == 'query-knowledge' and set(action) == {'action', 'query'}:
            result = self._action('knowledge', lambda: self.knowledge(action['query'])
                                  if self.knowledge else {'status': 'unavailable'})
        else:
            raise ValueError('invalid tool action')
        record = self._artifact(action['action'], result)
        self.state['history'].append({'tool': action, 'result': record, 'content': result})
        self._save()
