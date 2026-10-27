"""Matched transfer trials and an independent evidence-reading memory judge.

Execution callbacks must use fresh Actors, disable learning publication, and
observe retained external operations when re-entering the same arm directory.
The evaluator never admits memories or controls rewrite counts.
Completed judgments, including inconclusive, are immutable on resume. An arm
whose callback was interrupted is held; retrying requires a fresh directory
and external certification that its original operations have settled.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .subject_execution import SubjectOutcome
from .project_verifier import bounded_review_messages, artifact_preview, artifact_page
from .time_budget import AttemptTimeBudget


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('elapsed allocation must be finite and positive')
    return value


def _inconclusive(reason):
    return dict(decision='inconclusive', reason=reason, evidence=[], instructions=None)


def _preview(value, depth=0):
    """Bound initial metadata; full immutable data stays available by hash."""
    if isinstance(value, str):
        return value.encode('utf-8')[:1024].decode('utf-8', errors='ignore')
    if isinstance(value, dict):
        return {str(key)[:128]: _preview(item, depth + 1) for key, item in list(value.items())[:16]} if depth < 2 else {'items': len(value)}
    if isinstance(value, (list, tuple)):
        return [_preview(item, depth + 1) for item in value[:8]] if depth < 2 else {'items': len(value)}
    return value


def _proposal_preview(proposal):
    result = {key: _preview(proposal[key]) for key in ('operation', 'base_versions', 'summary',
        'applicability', 'preconditions', 'expectations', 'limitations', 'before_unavailable_reason') if key in proposal}
    for role in ('before', 'after'):
        snippet = proposal.get(role)
        if isinstance(snippet, dict):
            result[role] = {key: snippet.get(key) for key in ('sha256', 'start_line', 'end_line')}
            result[role]['text_preview'] = str(snippet.get('text', '')).encode('utf-8')[:512].decode('utf-8', errors='ignore')
    return result


def _task_contract(task):
    """IDs and narrative alone do not make a fresh comparison workload."""
    if isinstance(task, dict):
        return {key: _task_contract(value) for key, value in task.items()
                if key not in {'project_id', 'hypothesis', 'repairs'}}
    if isinstance(task, list):
        return [_task_contract(value) for value in task]
    return task


class MemoryVerifierAgent:
    """The judge can inspect actual logs; a numeric tie does not force rejection."""
    def __init__(self, complete, *, clock=time.monotonic):
        self.complete, self.clock, self.last_trace = complete, clock, []

    def review(self, proposal, arms, read_artifact, *, max_seconds, check_remaining=None, history=()):
        deadline = self.clock() + _seconds(max_seconds)
        self.last_trace = []
        proposal_bytes = canonical(proposal)
        proposal_sha = _hash(proposal_bytes)
        local_bytes = {proposal_sha: proposal_bytes}
        registry = {'proposal': proposal_sha}
        known, required = {proposal_sha}, set()
        arm_context = {}
        for name, arm in arms.items():
            for role, digest in arm.artifacts.items():
                registry[name + ':' + role] = digest
                known.add(digest)
                if any(word in role.lower() for word in ('log', 'result', 'execution')):
                    required.add(digest)
            arm_context[name] = dict(host_verified=arm.host_verified, source_sha256=arm.source_sha256,
                artifacts=artifact_preview(arm.artifacts), artifact_count=len(arm.artifacts),
                observations={key: _preview(value) for key, value in arm.observations.items() if key in
                    {'classification', 'summary', 'action_counts', 'metrics', 'speedup', 'elapsed_seconds',
                     'tokens', 'iterations', 'acceptance_score', 'compile_count', 'run_count'}})
        for index, prior in enumerate(history):
            prior_bytes = canonical(prior)
            prior_sha = _hash(prior_bytes)
            local_bytes[prior_sha] = prior_bytes
            registry[f'history:{index}:evaluation'] = prior_sha
            required.add(prior_sha)
            for role, digest in prior.get('artifacts', {}).items():
                if role.startswith('history:'):
                    continue
                registry[f'history:{index}:{role}'] = digest
                if any(word in role.lower() for word in ('log', 'result', 'execution')):
                    required.add(digest)
        known.update(registry.values())
        read = set()
        messages = [{'role': 'system', 'content': (
            'You are an independent memory usefulness verifier. Inspect matched transfer '
            'trial metrics and retained Actor/tool logs through read_artifact. Compare '
            'control accepted memory with treatment candidate memory. Both succeeding '
            'does not prove uselessness: logs may establish more direct reasoning, better '
            'transfer, fewer mistakes or unnecessary work. Conversely, apparent numeric '
            'gains may be accidental. Cite actual observed evidence and explain why the '
            'memory likely contributed to useful behavior or needs improvement. One properly '
            'matched transfer pair with an explicit retained memory lookup/application and '
            'specific avoided control mistakes can justify acceptance as a qualitative '
            'inference, even when final metrics tie. Repeated trials or statistical '
            'significance are not prerequisites. Distinguish this inference from measured '
            'speedup. Prior failed candidate revisions followed by a verified final repair '
            'are valid evidence of process quality; final failed, incomplete or mismatched '
            'arms are inconclusive. Weak or counterproductive advice can warrant rewrite '
            'or rejection even if the final repaired treatment is fast. Host correctness is '
            'authoritative: never accept a failed treatment. Initial artifacts and proposal are previews. '
            'Inspect the full candidate with read_artifact using proposal_sha256 when needed. '
            'List admitted artifacts with {"list_artifacts":{"offset":0,"limit":20,"prefix":"control:"}}; '
            'prefix may be omitted or treatment:. Never add API metadata such as type or response_format. Return one JSON action: '
            '{"read_artifact":{"sha256":"...","offset":0,"limit":8192}} or '
            '{"decision":"accept|rewrite|reject|inconclusive","reason":"...",'
            '"evidence":["actually read artifact hashes"],"instructions":null|"concrete rewriting guidance"}. '
            'Inspect supplied result/log artifacts before any terminal judgment, including inconclusive. '
            'A rewrite requires concrete instructions; other decisions have null instructions.')},
            {'role': 'user', 'content': json.dumps(dict(proposal=_proposal_preview(proposal),
                proposal_sha256=proposal_sha, artifact_count=len(registry), arms=arm_context), sort_keys=True)}]
        messages[0]['content'] += (
            ' Retain unfavorable prior trials in your cumulative assessment. History includes prior '
            'candidate revisions; distinguish their proposal hashes from the current candidate. '
            'Invalid historical pairs do not establish benefit or a memory quality failure. '
            'Host-passing arms do not establish harmlessness or preclude rejection. If a VALID pair '
            'remains inconclusive after inspecting its evidence and prior results/logs, you may request '
            'a discriminating additional experiment by adding exactly '
            '"next_probe":{"question":"specific unresolved uncertainty and needed contrast"} '
            'to the inconclusive decision. Do not include code, an algorithm or implementation advice '
            'in that question; a separate curriculum selector will author the neutral host contract. '
            'A different project ID alone is not a new workload. Otherwise omit next_probe. '
            'No fixed number of probes is required; use existing evidence before requesting more.')
        context = json.loads(messages[1]['content'])
        context.update(history_count=len(history), history=[dict(index=i,
            evaluation_sha256=_hash(canonical(prior)), decision=prior.get('decision'),
            reason=_preview(prior.get('reason', '')), validity=prior.get('validity'),
            evaluation_id=prior.get('evaluation_id'))
            for i, prior in list(enumerate(history))[-8:]])
        messages[1]['content'] = json.dumps(context, sort_keys=True)

        def remaining():
            available = deadline - self.clock()
            if check_remaining is not None:
                cap = check_remaining()
                if cap is not None:
                    available = min(available, cap)
            if available <= 0:
                raise TimeoutError('memory judge elapsed allocation exhausted')
            return available

        self.last_trace.append(dict(kind='request', messages=messages.copy()))
        while True:
            working = bounded_review_messages(messages, dict(read_artifacts=sorted(read),
                required_artifacts=sorted(required),
                arm_host_verified={name: arm.host_verified for name, arm in arms.items()}))
            self.last_trace.append(dict(kind='provider-request', messages=working.copy()))
            raw = self.complete(tuple(working), remaining_seconds=remaining())
            remaining()
            self.last_trace.append(dict(kind='response', text=raw))
            messages.append(dict(role='assistant', content=raw))
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError('judge action must be an object')
                if set(value) == {'list_artifacts'}:
                    observation = artifact_page(registry, value['list_artifacts'])
                    self.last_trace.append(dict(kind='artifact-list', request=value['list_artifacts'], observation=observation))
                elif set(value) == {'read_artifact'}:
                    request = value['read_artifact']
                    if not isinstance(request, dict) or set(request) - {'sha256', 'offset', 'limit'}:
                        raise ValueError('invalid artifact read')
                    digest = request['sha256']
                    if digest not in known:
                        raise ValueError('artifact is not retained by the matched trial')
                    data = local_bytes[digest] if digest in local_bytes else read_artifact(digest, remaining_seconds=remaining())
                    if not isinstance(data, bytes) or _hash(data) != digest:
                        raise ValueError('artifact bytes do not match identity')
                    offset, limit = request.get('offset', 0), request.get('limit', 8192)
                    if type(offset) is not int or not 0 <= offset <= len(data) or type(limit) is not int or not 1 <= limit <= 65536:
                        raise ValueError('invalid artifact page')
                    limit = min(limit, 8192)
                    page = data[offset:offset + limit]
                    if page or not data:
                        read.add(digest)
                    observation = dict(sha256=digest, offset=offset, size=len(data),
                        text=page.decode('utf-8', errors='replace'),
                        next_offset=offset + limit if offset + limit < len(data) else None)
                    self.last_trace.append(dict(kind='read', request=request, observation=observation))
                else:
                    keys = {'decision', 'reason', 'evidence', 'instructions'}
                    if set(value) not in (keys, keys | {'next_probe'}) or value['decision'] not in {'accept', 'rewrite', 'reject', 'inconclusive'}:
                        raise ValueError('invalid memory judgment')
                    if not isinstance(value['reason'], str) or not value['reason'].strip():
                        raise ValueError('judgment requires an evidence-based reason')
                    evidence = value['evidence']
                    if not isinstance(evidence, list) or any(item not in read for item in evidence):
                        raise ValueError('judgment cites evidence not actually read')
                    if required - read:
                        raise ValueError('inspect missing required current and historical evidence before judgment')
                    if value['decision'] != 'inconclusive' and not evidence:
                        raise ValueError('substantive judgment requires actually read evidence references')
                    if value['decision'] == 'accept' and not arms['treatment'].host_verified:
                        raise ValueError('failed treatment cannot support memory acceptance')
                    if value['decision'] == 'rewrite':
                        if not isinstance(value['instructions'], str) or not value['instructions'].strip():
                            raise ValueError('rewrite requires concrete instructions')
                    elif value['instructions'] is not None:
                        raise ValueError('only rewrites carry instructions')
                    if 'next_probe' in value:
                        probe = value['next_probe']
                        if (value['decision'] != 'inconclusive' or not isinstance(probe, dict)
                                or set(probe) != {'question'} or not isinstance(probe['question'], str)
                                or not probe['question'].strip() or len(probe['question']) > 2048):
                            raise ValueError('next_probe requires an inconclusive decision and one concrete question')
                    remaining()
                    self.last_trace.append(dict(kind='decision', value=value))
                    return value
                remaining()
            except (ValueError, KeyError, TypeError) as error:
                observation = dict(error=str(error), expected_top_level=[['read_artifact'], ['list_artifacts'],
                    ['decision', 'reason', 'evidence', 'instructions'],
                    ['decision', 'reason', 'evidence', 'instructions', 'next_probe']],
                    correction='Return only one exact key set; remove API metadata and unknown keys.')
                missing = required - read
                if missing:
                    observation.update(missing_required_count=len(missing),
                        missing_required=[dict(sha256=sha,
                            roles=[role[:160] for role, digest in sorted(registry.items()) if digest == sha][:4])
                            for sha in sorted(missing)[:16]],
                        correction='Read the listed missing hashes. Current and historical result/log evidence is required; preserve your observations already read.')
                    if any(role.startswith('history:') and sha in missing for role, sha in registry.items()):
                        observation['discovery'] = dict(list_artifacts=dict(prefix='history:', offset=0, limit=20))
                self.last_trace.append(dict(kind='feedback', observation=observation))
            messages.append(dict(role='user', content=json.dumps(observation, sort_keys=True)))


class MemoryVerifier:
    def __init__(self, complete, select_trial, execute_arm, *, model_identity,
                 runtime_identity, max_seconds=1800, arm_seconds=600,
                 clock=time.monotonic, wall_clock=time.time):
        self.select_trial, self.execute_arm = select_trial, execute_arm
        self.judge = MemoryVerifierAgent(complete, clock=clock)
        self.model, self.runtime = model_identity, runtime_identity
        self.seconds, self.arm_seconds = _seconds(max_seconds), _seconds(arm_seconds)
        self.clock, self.wall_clock, self.last_trace = clock, wall_clock, []

    @contextmanager
    def _lock(self, directory):
        with (directory / '.evaluation.lock').open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def evaluate(self, proposal, baseline_bytes, treatment_bytes, directory, *,
                 remaining_seconds, task=None, sequence=0, check_remaining=None,
                 history=(), probe_request=None, prior_task=None, read_history_artifact=None):
        parent_deadline = self.clock() + _seconds(remaining_seconds)
        if not isinstance(baseline_bytes, bytes) or not isinstance(treatment_bytes, bytes):
            raise TypeError('paired memories must be immutable bytes')
        if type(sequence) is not int or sequence < 0:
            raise ValueError('sequence must be nonnegative')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(proposal_sha256=_hash(canonical(proposal)), baseline_sha256=_hash(baseline_bytes),
            treatment_sha256=_hash(treatment_bytes), model=self.model, runtime=self.runtime,
            max_seconds=self.seconds, arm_seconds=self.arm_seconds, sequence=sequence)
        identity = json.loads(canonical(identity))
        if history or probe_request is not None:
            identity.update(history_sha256=[_hash(canonical(row)) for row in history],
                            probe_request=probe_request, prior_task=prior_task)
        path = directory / 'evaluation.json'
        with self._lock(directory):
            state = json.loads(path.read_bytes()) if path.exists() else dict(
                identity=identity, allocation=min(self.seconds, remaining_seconds), arms={}, trace=[], task=None)
            if state['identity'] != identity or task is not None and state['task'] is not None and state['task'] != task:
                raise ValueError('matched trial identity changed')
            if state.get('decision') is not None:
                self.last_trace = state['trace']
                return state['decision']
            # Persist allocation/identity before any budget interval or callback.
            atomic(path, canonical(state))
            store = ContentAddressedArtifacts(directory / 'artifacts')
            store.put_bytes(canonical(proposal))
            history_refs = {}
            for index, prior in enumerate(history):
                history_refs[f'history:{index}:evaluation'] = store.put_bytes(canonical(prior))
                refs = {role: digest for role, digest in prior.get('artifacts', {}).items()
                        if not role.startswith('history:')}
                history_refs.update({f'history:{index}:{role}': digest for role, digest in refs.items()})
                for digest in set(refs.values()):
                    if read_history_artifact is not None:
                        if store.put_bytes(read_history_artifact(digest)) != digest:
                            raise ValueError('prior trial artifact identity changed')
                    else:
                        store.read_bytes(digest)
            for role, data in (('control', baseline_bytes), ('treatment', treatment_bytes)):
                store.put_bytes(data)
            budget = AttemptTimeBudget(directory / 'elapsed.json', state['allocation'],
                                       clock=self.clock, wall_clock=self.wall_clock)

            def save():
                self.last_trace = state['trace']
                atomic(path, canonical(state))

            def remaining():
                available = min(parent_deadline, budget.deadline) - self.clock()
                if check_remaining is not None:
                    cap = check_remaining()
                    if cap is not None:
                        available = min(available, cap)
                if available <= 0:
                    raise TimeoutError('matched trial elapsed allocation exhausted')
                return available

            try:
                if state['task'] is None:
                    options = dict(remaining_seconds=remaining())
                    if probe_request is not None:
                        options.update(probe_request=probe_request, prior_task=prior_task, history=history)
                    selected = task if task is not None else self.select_trial(proposal, **options)
                    remaining()
                    if not isinstance(selected, dict) or not selected:
                        raise ValueError('select exactly one nonempty transfer task')
                    frozen = json.loads(canonical(selected))
                    if probe_request is not None and any(_task_contract(frozen) == _task_contract(row.get('trial_task')) for row in history):
                        raise ValueError('follow-up selector repeated a prior workload contract')
                    allocation = min(self.arm_seconds, remaining() / 3)
                    state['task'], state['arm_allocation'] = frozen, allocation
                    state['order'] = ['control', 'treatment'] if sequence % 2 == 0 else ['treatment', 'control']
                    state['trace'].append(dict(kind='task-frozen', task=state['task'],
                        arm_seconds=state['arm_allocation'], order=state['order']))
                    save()
                for name in state['order']:
                    # An invalid/interrupted arm may still own external work.
                    # Hold the entire pair, including its unstarted peer.
                    if any(not row.get('valid') or row.get('outcome') is None
                           for row in state['arms'].values()):
                        break
                    if name in state['arms']:
                        continue
                    arm_dir = directory / name
                    arm_dir.mkdir(exist_ok=True)
                    arm_budget = AttemptTimeBudget(arm_dir / 'elapsed.json', state['arm_allocation'],
                                                  clock=self.clock, wall_clock=self.wall_clock)
                    try:
                        allowance = arm_budget.deadline - self.clock()
                        if allowance <= 0 or remaining() < allowance:
                            raise TimeoutError('insufficient remaining matched arm allocation')
                        state['trace'].append(dict(kind='arm-started', arm=name,
                            memory_sha256=_hash(treatment_bytes if name == 'treatment' else baseline_bytes),
                            remaining_seconds=allowance))
                        # An interrupted callback may already own external work.
                        # Resume must hold, not dispatch it again without settlement.
                        state['arms'][name] = dict(outcome=None, valid=False, error_type='InterruptedArm')
                        save()
                        outcome = self.execute_arm(json.loads(canonical(state['task'])), treatment_bytes if name == 'treatment' else baseline_bytes,
                                                   arm_dir, remaining_seconds=allowance)
                        if not isinstance(outcome, SubjectOutcome) or type(outcome.host_verified) is not bool:
                            raise TypeError('arm must return a host-owned SubjectOutcome')
                        for digest in outcome.artifacts.values():
                            store.put_bytes(outcome.read_artifact(digest))
                        status = outcome.observations
                        evidence = status.get('evidence', {})
                        statuses = [status.get('classification'), status.get('status'), status.get('actor_status')]
                        if isinstance(evidence, dict):
                            statuses += [evidence.get('classification'), evidence.get('actor_status')]
                        invalid = any(any(word in str(item).lower().replace('_', '-') for word in
                            ('infrastructure', 'provider', 'timeout', 'timed-out', 'time-budget',
                             'budget-exhausted', 'cancelled', 'interrupted', 'context-budget')) for item in statuses)
                        valid = self.clock() <= arm_budget.deadline and not invalid
                        state['arms'][name] = dict(outcome=outcome.as_dict(), valid=valid,
                            error_type=None if valid else 'InvalidArmStatus' if invalid else 'TimeoutError')
                        state['trace'].append(dict(kind='arm-completed', arm=name, **state['arms'][name]))
                    except Exception as error:
                        state['arms'][name] = dict(outcome=None, valid=False, error_type=type(error).__name__)
                        state['trace'].append(dict(kind='arm-inconclusive', arm=name, error_type=type(error).__name__))
                    finally:
                        arm_budget.close()
                        save()
                    # Preserve completed evidence before honoring a parent stop;
                    # never start its peer or a judge after cancellation.
                    remaining()
                    if not state['arms'][name]['valid']:
                        break
                validity = {key: value.get('valid', False) for key, value in state['arms'].items()}
                if set(validity) != {'control', 'treatment'} or not all(validity.values()):
                    decision = _inconclusive('Incomplete, late or infrastructure-failed arms cannot establish memory benefit.')
                else:
                    arms = {name: SubjectOutcome(**row['outcome'], artifact_root=store.root)
                            for name, row in state['arms'].items()}
                    self.judge.last_trace = []
                    try:
                        decision = self.judge.review(proposal, arms,
                            lambda digest, **kw: store.read_bytes(digest), max_seconds=remaining(),
                            check_remaining=remaining, history=history)
                    finally:
                        state['trace'].extend(self.judge.last_trace)
                remaining()
                decision.update(evaluation_id=_hash(canonical(identity)), trial_task=state['task'], validity=validity,
                    arms={name: row['outcome'] for name, row in state['arms'].items()},
                    artifacts={name + ':' + role: digest for name, row in state['arms'].items()
                               for role, digest in (row['outcome'] or {}).get('artifacts', {}).items()})
                decision['artifacts']['proposal'] = identity['proposal_sha256']
                decision['artifacts'].update(history_refs)
                state['decision'] = decision
                save()
                return decision
            except Exception as error:
                decision = _inconclusive('Matched trial or judge did not complete; inspect retained evidence before resuming.')
                decision['error_type'] = type(error).__name__
                decision.update(evaluation_id=_hash(canonical(identity)), trial_task=state['task'],
                                validity={key: row.get('valid', False) for key, row in state['arms'].items()},
                                arms={name: row['outcome'] for name, row in state['arms'].items()},
                                artifacts={name + ':' + role: digest for name, row in state['arms'].items()
                                           for role, digest in (row['outcome'] or {}).get('artifacts', {}).items()})
                decision['artifacts']['proposal'] = identity['proposal_sha256']
                decision['artifacts'].update(history_refs)
                state['decision'] = decision
                save()
                return decision
            finally:
                budget.close()
                save()
