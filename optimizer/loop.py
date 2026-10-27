"""Durable backend-neutral optimizer. Backends own retained-operation recovery.

prepare(spec, directory) freezes reference/inputs and returns suite_sha256 plus
cases [{case, contract, timing:{torch:{latency_us}}}]. evaluate(source,directory,
block_count=) returns native.evaluate JSON (profile=False, benchmark=True).
profile(source,directory,block_count=) returns compact diagnostics. Every backend
call MUST be idempotent for its directory and observe existing handles on retry.
"""
from __future__ import annotations

import copy
import base64
import json
import math
import re
import time

from core.knowledge_presentation import compact_knowledge_result

from .cases import validate_submission, normalize_job_submission
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
    def __init__(self, store, job_id, agent, backend, *, knowledge=None, clock=time.time,
                 max_cases=24):
        if type(max_cases) is not int or not 1 <= max_cases <= 24:
            raise ValueError('max_cases must be between one and 24')
        self.max_cases = max_cases
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
        return row['status'] == 'running' or (
            row['status'] == 'budget-exhausted' and 'pending' in self.state)

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
            if stage in {'cases', 'optimize'}:
                self.state['model_action'] = self.state['action_result']
            return value
        finally:
            self.store.account_time(self.key, max(0, self.clock() - started))
            self.state.pop('inflight_started', None)
            self._save()

    def _spec(self, submission):
        return normalize_job_submission(submission)

    def run(self, *, prepare_only=False, single_evaluation=False, memory_snapshot=None, context=None):
        try:
            from benchmarks.ascend.actor_control import ActorControl
            with ActorControl(self.directory.parent).lease():
                return self._run_owned(prepare_only=prepare_only, single_evaluation=single_evaluation,
                                 memory_snapshot=memory_snapshot, context=context)
        finally:
            close = getattr(self.agent, "close", None)
            if close is not None:
                close()

    def _run_owned(self, *, prepare_only=False, single_evaluation=False, memory_snapshot=None, context=None):
        """Optionally stop at Phase 2's preparation or single-evaluation boundary.

        These boundaries reuse this job's durable state, reference and pending
        native operation. They do not create a fresh optimizer search.
        """
        self.prepare_only, self.single_evaluation = prepare_only, single_evaluation
        self.context = context or {}
        row = self.store.get(self.key)
        if row['status'] not in {'queued', 'running'} and not (
                row['status'] == 'budget-exhausted' and 'pending' in self.state):
            return row
        submission = self.store.submission(self.key)
        verify_snapshot(submission['memory_snapshot'])
        if memory_snapshot is not None:
            verify_snapshot(memory_snapshot)
            submission = {**submission, 'memory_snapshot': memory_snapshot}
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
            status = self.store.get(self.key)['status']
            if status in {'running', 'budget-exhausted'}:
                self.store.transition(self.key, status, stage='observation-unavailable')
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
        if len(spec['signatures']) > self.max_cases:
            raise ValueError('supplied representatives exceed configured case budget')
        if 'spec' not in self.state:
            action = self._model('cases', {'submission': spec, 'max_cases': self.max_cases,
                                         'remaining_signatures': self.max_cases - len(spec['signatures'])})
            if action is None:
                return
            if set(action) != {'action', 'signatures', 'summary'} or action['action'] != 'cases':
                raise ValueError('first action must propose additional signatures')
            additions = action['signatures']
            if not isinstance(additions, list):
                raise ValueError('case proposals must be a list')
            accepted, rejected = [], []
            for signature in additions:
                try:
                    validate_submission({**spec, 'signatures': [signature]})
                    if len(spec['signatures']) + len(accepted) >= self.max_cases:
                        raise ValueError(f'case proposal exceeds configured {self.max_cases}-case budget')
                    accepted.append(signature)
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append({'signature': signature, 'error': str(exc)})
            self.store.add_event(self.key, 'case-proposals', {'accepted': accepted, 'rejected': rejected,
                                                           'summary': action['summary']})
            spec['signatures'] += accepted
            self.state['spec'] = self._artifact('frozen-reference-submission', spec)
            self.state.pop('action_result', None)
            self.state.pop('model_action', None)
            self._save()
        spec = self._read(self.state['spec'])
        if len(spec['signatures']) > self.max_cases:
            raise ValueError('frozen representatives exceed configured case budget')
        if 'baseline' not in self.state:
            baseline = self._action('prepare', lambda: self.backend.prepare(spec, self.directory / 'baseline'))
            if baseline is None:
                return
            if not baseline.get('suite_sha256') or not baseline.get('cases'):
                raise ValueError('backend must return frozen nonempty baseline')
            self.state['baseline'] = self._artifact('baseline', baseline)
            self._save()
        baseline = self._read(self.state['baseline'])
        if self.prepare_only:
            return
        while self._boundary():
            pending = self.state.get('pending')
            if pending is None:
                action = self._model('optimize', {'submission': spec, 'baseline': baseline,
                        'history': self.state['history'], 'best': self.state.get('best'),
                        'memory_summary': [{'index': i, 'summary': str(entry)[:1200]} for i, entry
                            in enumerate(submission['memory_snapshot']['memory_entries'])]})
                if action is None:
                    break
                if not self._boundary():
                    break
                kind = action.get('action')
                if kind == 'finish' and set(action) == {'action', 'summary'}:
                    self.store.add_event(self.key, 'finished', action)
                    if self.single_evaluation:
                        self.state.pop('action_result', None)
                        self.state.pop('model_action', None)
                        self._save()
                        return
                    self.store.transition(self.key, 'completed')
                    break
                if kind in {'query-knowledge', 'read-memory', 'read-artifact', 'compile'}:
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
                    self.state.pop('action_result', None)
                    self.state.pop('model_action', None)
                    self._save()
                    continue
                pending = {**action, 'sha256': source_sha, 'configuration_sha256': identity,
                           'attempt': self.store.get(self.key)['attempts'] + 1}
                self.state['pending'] = pending
                self.state['candidate'] = self._artifact('candidate.cpp', action['source'])
                self.state.pop('action_result', None)
                self.state.pop('model_action', None)
                self._save()
            if not self._boundary():
                break
            # Attempt is reserved before submission: crashes never lose its charge.
            if self.store.get(self.key)['attempts'] < pending['attempt']:
                self.store.record_attempt(self.key, candidate_sha256=pending['sha256'], result='evaluating')
            # Reconcile an already retained attempt even after its time allocation expires.
            if 'evaluation' not in pending:
                result = self._evaluate_reserved(pending)
                pending['evaluation'] = self._artifact('evaluation', result)
                self._save()
            result = self._read(pending['evaluation'])
            value = score(baseline, result)
            history = {'candidate_sha256': pending['sha256'], 'result': pending['evaluation'],
                       'source': self.state['candidate'],
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
                    except ObservationUnavailable:
                        raise
                    except WorkerStopped:
                        row = self.store.get(self.key)
                        if (row['stop_requested'] or (row['elapsed_seconds'] < row['max_seconds']
                                and time.monotonic() < getattr(self.backend, 'deadline', float('inf')))):
                            raise
                        # Correctness/evaluation is already committed. Optional
                        # diagnostics exhausting the deadline cannot erase it or
                        # leave a zero-budget pending profile retry forever.
                        self.store.add_event(self.key, 'profiling-unavailable',
                            {'reason': 'wall-time budget exhausted after evaluation',
                             'candidate_sha256': pending['sha256']})
                    except Exception as exc:
                        self.store.add_event(self.key, 'profiling-unavailable', {'error': str(exc)})
                else:
                    self.store.add_event(self.key, 'profiling-unavailable',
                                         {'reason': 'budget or stop boundary', 'candidate_sha256': pending['sha256']})
            self.state.pop('pending')
            self._save()
            if self.single_evaluation:
                return

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
        original_context = context
        from benchmarks.ascend.actor_control import ActorControl, ActorStopped
        control = ActorControl(self.directory.parent)
        retained = self.state.get('model_action', self.state.get('action_result'))
        if retained and retained.get('name') == stage and not control.pending():
            # The caller removes this result in the same checkpoint that applies
            # it. A crash in between must not issue another billable request.
            return self._read(retained)
        if retained and retained.get('name') == stage:
            self.state.pop('model_action', None)
            self.state.pop('action_result', None)
        self.state['turn'] += 1
        self._save()
        context = {**context, **self.context}
        checkpoint = control.status()['checkpoint'] or {}
        feedback = checkpoint.get('feedback', self.state.get('user_feedback', []))
        self.state['user_feedback'] = feedback
        try:
            control.boundary({'turn': self.state['turn'], 'feedback': feedback}, feedback)
        except ActorStopped:
            row = self.store.stop(self.key)
            if row['status'] == 'running':
                self.store.transition(self.key, 'stopped')
            return None
        self._save()
        context['user_feedback'] = feedback
        # History can be paged as a whole by the coding context layer. Keep the
        # latest bounded knowledge response readable independently of that list.
        for entry in reversed(self.state['history']):
            if entry.get('tool', {}).get('action') == 'query-knowledge':
                knowledge = entry.get('content')
                if (isinstance(knowledge, dict)
                        and knowledge.get('event') == 'knowledge-query'
                        and 'results_index_sha256' in knowledge
                        and len(canonical(knowledge)) <= 3500):
                    context['latest_knowledge'] = knowledge
                break
        row = self.store.get(self.key)
        context['remaining_seconds'] = max(0, row['max_seconds'] - row['elapsed_seconds'])
        if self.agent.identity.get('coding_engine') == 'pi':
            context['knowledge_available'] = self.knowledge is not None
        if (self.single_evaluation or self.prepare_only or
                any(item.get('content', {}).get('failure_case') for item in context.get('history', []))):
            # Keep every lesson addressable, but don't replay hundreds of full
            # entries or accumulated compiler/profiler logs on every turn.
            rows = context.get('memory_summary', [])
            limit = max(16, min(1200, 16000 // max(1, len(rows))))
            context['memory_summary'] = [{**row, 'summary': row['summary'][:limit]} for row in rows]
            history = []
            recent = context.get('history', [])[-12:]
            for index, row in enumerate(recent):
                record = {key: value for key, value in row.items() if key != 'content'}
                if len(json.dumps(record, ensure_ascii=True)) > 3000:
                    record = {'metadata_artifact': self._artifact('history-entry', row),
                              'result': row.get('result'), 'summary': str(row.get('summary', ''))[:500]}
                content = json.dumps(row.get('content'), ensure_ascii=True)
                size = 12000 if index == len(recent) - 1 and 'tool' in row else 1500
                if (row.get('tool', {}).get('action') == 'query-knowledge'
                        and isinstance(row.get('content'), dict)
                        and row['content'].get('event') == 'knowledge-query'
                        and 'results_index_sha256' in row['content']
                        and len(canonical(row['content'])) <= 3500):
                    size = len(content)
                record['content_preview'] = content[:size]
                record['content_truncated'] = len(content) > size
                history.append(record)
            context['history'] = history
            if 'baseline' in context:
                baseline = dict(context['baseline'])
                profiling = baseline.get('profiling')
                if profiling is not None and len(json.dumps(profiling, ensure_ascii=True)) > 12000:
                    baseline['profiling'] = {'artifact': self._artifact('baseline-profiling', profiling),
                        'preview': json.dumps(profiling, ensure_ascii=True)[:1500], 'truncated': True}
                context['baseline'] = baseline
            for key in ('best', 'phase2'):
                if key in context and len(rendered := json.dumps(context[key], ensure_ascii=True)) > 12000:
                    record = self._artifact('context-' + key, context[key])
                    context[key] = {'artifact': record, 'preview': rendered[:12000], 'truncated': True}
        from benchmarks.ascend.status_descriptions import describe_statuses
        presentation = describe_statuses(context)
        value = self._action(stage, lambda: self.agent.complete(stage, presentation, self._record))
        if value is not None and control.pending():
            # A redirection arriving in flight supersedes this unexecuted action.
            self.state.pop('model_action', None)
            self.state.pop('action_result', None)
            self._save()
            return self._model(stage, original_context)
        return value

    def _tool(self, action, snapshot):
        # Model request mistakes are feedback, not evidence/transport failures.
        # Validate before accessing storage so corruption exceptions still escape.
        error = None
        if action['action'] == 'read-artifact':
            if set(action) != {'action', 'sha256', 'offset'}:
                error = 'read-artifact requires exactly action, sha256 and offset'
            elif not isinstance(action['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', action['sha256']):
                error = 'sha256 must be the exact 64-character lowercase hexadecimal artifact digest; do not abbreviate or guess'
            elif type(action['offset']) is not int or action['offset'] < 0:
                error = 'offset must be a nonnegative integer'
        elif action['action'] == 'read-memory':
            if set(action) != {'action', 'index'}:
                error = 'read-memory requires exactly action and index'
            elif type(action['index']) is not int or not 0 <= action['index'] < len(snapshot['memory_entries']):
                error = 'index must be an integer listed in memory_summary'
        if error:
            self._tool_result(action, {'status': 'tool-error', 'error': error})
            return
        if action['action'] == 'compile' and set(action) == {'action', 'source', 'block_count', 'summary'}:
            if (not isinstance(action['source'], str) or not action['source'].strip()
                    or type(action['block_count']) is not int or not 1 <= action['block_count'] <= 65535):
                raise ValueError('invalid compile action')
            identity = sha(canonical({'source': action['source'], 'block_count': action['block_count']}))
            result = self._action('compile', lambda: self.backend.compile(action['source'],
                self.directory / 'compile' / identity, block_count=action['block_count']))
        elif action['action'] == 'read-artifact' and set(action) == {'action', 'sha256', 'offset'}:
            offset = action['offset']
            if type(offset) is not int or offset < 0:
                raise ValueError('invalid artifact offset')
            try:
                data = self.store.read_artifact(self.key, action['sha256'])
            except FileNotFoundError:
                if any(row['sha256'] == action['sha256'] for row in self.store.artifacts(self.key)):
                    # A recorded artifact disappearing is evidence loss, not an
                    # unknown reference invented by the model.
                    raise
                record = next((row for row in snapshot['files'].values()
                               if row['sha256'] == action['sha256']), None)
                if record is None:
                    self._tool_result(action, {'status': 'tool-error', 'error':
                        'artifact is absent from job and authenticated memory snapshot; use an exact retained reference or read-memory index'})
                    return
                data = base64.b64decode(record['base64'], validate=True)
            raw = data.decode('utf-8')
            result = {'offset': offset, 'text': raw[offset:offset + 1500],
                      'next_offset': offset + 1500 if len(raw) > offset + 1500 else None}
        elif action['action'] == 'read-memory' and set(action) == {'action', 'index'}:
            index = action['index']
            if type(index) is not int or not 0 <= index < len(snapshot['memory_entries']):
                raise ValueError('invalid memory index')
            result = snapshot['memory_entries'][index]
        elif action['action'] == 'query-knowledge' and set(action) == {'action', 'query'}:
            def query():
                raw = self.knowledge(action['query']) if self.knowledge else {'status': 'unavailable'}
                return compact_knowledge_result(action['query'], raw, lambda content:
                    self.store.put_artifact(self.key, 'knowledge-source', content)['sha256'])

            result = self._action('knowledge', query)
        else:
            raise ValueError('invalid tool action')
        self._tool_result(action, result)

    def _tool_result(self, action, result):
        record = self._artifact(action['action'], result)
        self.state['history'].append({'tool': action, 'result': record, 'content': result})
        self.state.pop('action_result', None)
        self.state.pop('model_action', None)
        self._save()
