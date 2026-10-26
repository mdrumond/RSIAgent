"""Matched execution, qualitative evidence reads and resumable elapsed limits."""
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.memory_verifier import MemoryVerifier
from benchmarks.ascend.subject_execution import SubjectOutcome


class Harness:
    def __init__(self, *, treatment_passes=True, control_passes=True, failure=None, misread=False, status=None):
        self.now = 0.0
        self.calls, self.selections, self.requests = [], [], []
        self.treatment_passes, self.failure = treatment_passes, failure
        self.control_passes = control_passes
        self.misread, self.status = misread, status
        self.judge_attempts = 0
        self.verifier = MemoryVerifier(self.complete, self.select, self.execute,
            model_identity={'model': 'same'}, runtime_identity={'target': 'synthetic'},
            max_seconds=100, arm_seconds=10, clock=lambda: self.now, wall_clock=lambda: self.now)

    def select(self, proposal, *, remaining_seconds):
        self.selections.append((proposal, remaining_seconds))
        return {'project_id': 'transfer', 'inputs': [3, -2]}

    def execute(self, task, memory, directory, *, remaining_seconds):
        name = Path(directory).name
        self.calls.append((name, memory, Path(directory), remaining_seconds, task))
        self.now += 1
        if self.failure == name:
            self.failure = None
            raise ConnectionError('observe the existing operation on retry')
        store = ContentAddressedArtifacts(Path(directory) / 'artifacts')
        passed = self.control_passes if name == 'control' else self.treatment_passes
        log = store.put_json(dict(mistakes=3 if name == 'control' else 2 if self.misread else 0,
            used_lesson=name == 'treatment', source='full Actor reasoning and tool log'))
        result = store.put_json(dict(host_verified=passed, speedup=1))
        source = store.put_bytes((name + ' candidate').encode())
        observations = dict(speedup=1, classification=self.status or ('host-verified' if passed else 'host-mismatch'))
        return SubjectOutcome(passed, source, {'source': source, 'log:actor': log, 'result': result}, observations, store.root)

    def complete(self, messages, *, remaining_seconds):
        self.requests.append((messages, remaining_seconds))
        context = json.loads(messages[1]['content'])
        read = {}
        for message in messages[2:]:
            if message['role'] == 'user':
                value = json.loads(message['content'])
                if 'sha256' in value:
                    read[value['sha256']] = value['text']
        artifacts = {name: arm['artifacts'] for name, arm in context['arms'].items()}
        for roles in artifacts.values():
            for role, digest in roles.items():
                if role != 'source' and digest not in read:
                    return json.dumps({'read_artifact': {'sha256': digest}})
        logs = {name: json.loads(read[roles['log:actor']]) for name, roles in artifacts.items()}
        treatment = context['arms']['treatment']
        # Deliberately try accepting a failed treatment once; the verifier must
        # reject that decision before this callback supplies a supported rewrite.
        accept = logs['treatment']['mistakes'] == 0
        if not treatment['host_verified']:
            self.judge_attempts += 1
            accept = self.judge_attempts == 1
        choice = 'accept' if accept else 'rewrite'
        return json.dumps(dict(decision=choice, reason=json.dumps(logs), evidence=list(read),
                              instructions=None if accept else 'Explain the missed cases and repair the example.'))


def evaluate(harness, root, **options):
    return harness.verifier.evaluate({'summary': 'increment lesson'}, b'old', b'new', root,
                                     remaining_seconds=60, **options)


def test_tied_metrics_allow_log_supported_acceptance_and_completed_arms_do_not_repeat(tmp_path):
    h = Harness()
    result = evaluate(h, tmp_path)
    assert result['decision'] == 'accept'
    assert result['arms']['control']['observations']['speedup'] == result['arms']['treatment']['observations']['speedup']
    assert [row[:2] for row in h.calls] == [('control', b'old'), ('treatment', b'new')]
    assert [row[3] for row in h.calls] == [10, 10]
    assert h.calls[0][2] != h.calls[1][2] and h.calls[0][4] == h.calls[1][4]
    expected_reads = {digest for arm in result['arms'].values()
                      for role, digest in arm['artifacts'].items() if role != 'source'}
    assert {event['observation']['sha256'] for event in h.verifier.last_trace
            if event['kind'] == 'read'} == expected_reads
    assert set(result['artifacts'].values()) >= set(result['evidence'])
    assert evaluate(h, tmp_path) == result
    assert len(h.calls) == 2 and len(h.selections) == 1


def test_counterbalanced_order_and_caller_frozen_task(tmp_path):
    h = Harness()
    task = {'project_id': 'frozen-transfer', 'inputs': [8, 9]}
    result = evaluate(h, tmp_path, sequence=1, task=task)
    assert result['trial_task'] == task and h.selections == []
    assert [row[0] for row in h.calls] == ['treatment', 'control']
    with pytest.raises(ValueError, match='identity changed'):
        evaluate(h, tmp_path, sequence=1, task={'project_id': 'different'})
    assert len(h.calls) == 2


def test_failed_treatment_cannot_be_accepted_and_weaker_logs_request_rewrite(tmp_path):
    h = Harness(treatment_passes=False)
    result = evaluate(h, tmp_path / 'failed')
    assert result['decision'] == 'rewrite' and result['instructions']
    assert h.judge_attempts == 2
    assert any(event['kind'] == 'feedback' for event in h.verifier.last_trace)
    other = Harness(misread=True)
    assert evaluate(other, tmp_path / 'unhelpful')['decision'] == 'rewrite'


def test_transport_failure_is_cached_and_resume_never_reenters_unknown_arm(tmp_path):
    h = Harness(failure='treatment')
    first = evaluate(h, tmp_path)
    assert first['decision'] == 'inconclusive' and h.requests == []
    assert first['arms']['control'] is not None and first['arms']['treatment'] is None
    second = evaluate(h, tmp_path, task=first['trial_task'])
    assert second == first
    assert [row[0] for row in h.calls] == ['control', 'treatment']
    assert len(h.selections) == 1


@pytest.mark.parametrize('status', ['time-budget-exhausted', 'provider-failure', 'infrastructure-failure'])
def test_invalid_status_cannot_support_benefit_even_if_host_flag_is_true(tmp_path, status):
    h = Harness(status=status)
    result = evaluate(h, tmp_path)
    assert result['decision'] == 'inconclusive'
    assert not any(result['validity'].values())
    assert h.requests == []
    expected = {digest for arm in result['arms'].values() for digest in arm['artifacts'].values()}
    assert expected <= set(result['artifacts'].values())
    retained = ContentAddressedArtifacts(tmp_path / 'artifacts')
    assert json.loads(retained.read_bytes(result['artifacts']['proposal'])) == {'summary': 'increment lesson'}


def test_late_arm_is_inconclusive_and_does_not_replenish_total_allocation(tmp_path):
    h = Harness()
    original = h.verifier.execute_arm
    def late(*args, **kwargs):
        result = original(*args, **kwargs)
        h.now += 70
        return result
    h.verifier.execute_arm = late
    result = evaluate(h, tmp_path)
    assert result['decision'] == 'inconclusive'
    assert h.requests == []
    completed = len(h.calls)
    assert evaluate(h, tmp_path)['decision'] == 'inconclusive'
    assert len(h.calls) == completed
    assert json.loads((tmp_path / 'elapsed.json').read_bytes())['elapsed_seconds'] >= 71


def test_semantic_control_failure_and_verified_treatment_can_demonstrate_gain(tmp_path):
    h = Harness(control_passes=False)
    result = evaluate(h, tmp_path)
    assert result['decision'] == 'accept'
    assert result['validity'] == {'control': True, 'treatment': True}
    assert not result['arms']['control']['host_verified']
    assert result['arms']['treatment']['host_verified']


def test_late_judge_cannot_publish_acceptance_and_changed_memories_cannot_resume(tmp_path):
    h = Harness()
    original = h.verifier.judge.complete
    def late(messages, *, remaining_seconds):
        value = original(messages, remaining_seconds=remaining_seconds)
        h.now += remaining_seconds + 1
        return value
    h.verifier.judge.complete = late
    result = evaluate(h, tmp_path)
    assert result['decision'] == 'inconclusive'
    assert len(h.calls) == 2
    with pytest.raises(ValueError, match='identity changed'):
        h.verifier.evaluate({'summary': 'increment lesson'}, b'changed', b'new', tmp_path, remaining_seconds=60)
    assert len(h.calls) == 2


def test_invalid_trial_proposal_does_not_leave_half_frozen_resume_state(tmp_path):
    h = Harness()
    h.verifier.select_trial = lambda *args, **kwargs: []
    first = evaluate(h, tmp_path)
    assert first['decision'] == 'inconclusive' and first['trial_task'] is None
    assert h.calls == []
    h.verifier.select_trial = h.select
    assert evaluate(h, tmp_path) == first
    assert evaluate(h, tmp_path / 'new-trial')['decision'] == 'accept'
    assert len(h.selections) == 1


def test_large_judge_pages_fold_context_and_preserve_observed_evidence(tmp_path):
    from benchmarks.ascend.memory_verifier import MemoryVerifierAgent
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    hashes = [store.put_bytes((str(i) * 16384).encode()) for i in range(8)]
    arms = {name: SubjectOutcome(True, None,
        {f'log:page-{index}': digest for index, digest in enumerate(hashes[start:start + 4])},
        {'speedup': 1}, store.root) for name, start in [('control', 0), ('treatment', 4)]}
    malformed = dict(read_artifact={'sha256': hashes[0], 'limit': 65536}, type='json_object')
    responses = iter([malformed,
        *[{'read_artifact': {'sha256': digest, 'limit': 65536}} for digest in hashes],
        dict(decision='accept', reason='Both retained Actor logs establish direct transfer.',
             evidence=hashes, instructions=None)])
    sizes, folded = [], []
    def complete(messages, *, remaining_seconds):
        sizes.append(len(json.dumps(messages, ensure_ascii=False).encode()))
        assert sizes[-1] <= 40000
        for message in messages:
            if message['role'] == 'user':
                value = json.loads(message['content'])
                if 'retained_review_state' in value:
                    folded.append(value['retained_review_state'])
        return json.dumps(next(responses))
    agent = MemoryVerifierAgent(complete)
    result = agent.review({'summary': 'tail lesson'}, arms,
                          lambda digest, **kw: store.read_bytes(digest), max_seconds=60)
    assert result['decision'] == 'accept' and result['evidence'] == hashes
    assert folded and hashes[0] in folded[-1]['read_artifacts']
    assert folded[-1]['arm_host_verified'] == {'control': True, 'treatment': True}
    pages = [event['observation'] for event in agent.last_trace if event['kind'] == 'read']
    assert {page['sha256'] for page in pages} == set(hashes)
    assert all(len(page['text'].encode()) == 8192 and page['next_offset'] == 8192 for page in pages)
    feedback = [event['observation'] for event in agent.last_trace if event['kind'] == 'feedback']
    assert feedback[0]['expected_top_level'] == [['read_artifact'], ['list_artifacts'], ['decision', 'reason', 'evidence', 'instructions'], ['decision', 'reason', 'evidence', 'instructions', 'next_probe']]


def test_parent_stop_after_completed_arm_prevents_peer_and_judge(tmp_path):
    h = Harness()
    stopped = [False]
    original = h.verifier.execute_arm
    def execute(*args, **kwargs):
        outcome = original(*args, **kwargs)
        stopped[0] = True
        return outcome
    def check_remaining():
        if stopped[0]:
            raise TimeoutError('parent stopped')
        return 30
    h.verifier.execute_arm = execute
    result = h.verifier.evaluate({'summary': 'increment lesson'}, b'old', b'new', tmp_path,
        remaining_seconds=60, check_remaining=check_remaining)
    assert result['decision'] == 'inconclusive'
    assert [row[0] for row in h.calls] == ['control'] and h.requests == []
    assert result['arms']['control']['host_verified']
    assert result['validity'] == {'control': True}


def test_parent_stop_during_judge_prevents_further_requests_or_acceptance(tmp_path):
    h = Harness()
    stopped = [False]
    original = h.verifier.judge.complete
    def complete(messages, *, remaining_seconds):
        result = original(messages, remaining_seconds=remaining_seconds)
        stopped[0] = True
        return result
    def check_remaining():
        if stopped[0]:
            raise TimeoutError('parent stopped')
        return 30
    h.verifier.judge.complete = complete
    result = h.verifier.evaluate({'summary': 'increment lesson'}, b'old', b'new', tmp_path,
        remaining_seconds=60, check_remaining=check_remaining)
    assert result['decision'] == 'inconclusive'
    assert len(h.requests) == 1 and len(h.calls) == 2
    assert not any(event['kind'] == 'read' for event in h.verifier.last_trace)


def test_inconclusive_judge_must_inspect_valid_matched_evidence_before_finishing(tmp_path):
    from benchmarks.ascend.memory_verifier import MemoryVerifierAgent
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    digest = store.put_bytes(b'Equal final metrics; ambiguous transfer logs')
    arms = {name: SubjectOutcome(True, None, {'log:actor': digest}, {}, store.root)
            for name in ('control', 'treatment')}
    terminal = dict(decision='inconclusive', reason='The inspected pair does not distinguish benefit.',
                    evidence=[digest], instructions=None)
    replies = iter([dict(terminal, evidence=[]), {'read_artifact': {'sha256': digest}}, terminal])
    agent = MemoryVerifierAgent(lambda *args, **kwargs: json.dumps(next(replies)))
    result = agent.review({}, arms, lambda value, **kw: store.read_bytes(value), max_seconds=60)
    assert result == terminal
    assert len([event for event in agent.last_trace if event['kind'] == 'feedback']) == 1
    assert [event['observation']['sha256'] for event in agent.last_trace if event['kind'] == 'read'] == [digest]


def test_large_native_arm_registries_and_proposal_stay_paged_and_readable(tmp_path):
    from benchmarks.ascend.memory_verifier import MemoryVerifierAgent
    from optimizer.jobs import canonical
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    arms, registries = {}, {}
    for name in ('control', 'treatment'):
        registry = {f'source:prior-{index:04d}': store.put_bytes(f'{name} source candidate {index}'.encode())
                    for index in range(500)}
        registry['source'] = registry['source:prior-0000']
        registry['log:actor'] = store.put_json({'arm': name, 'mistakes': 3 if name == 'control' else 0})
        registry['result:execution'] = store.put_json({'passed': True, 'score': 100})
        registries[name] = registry
        arms[name] = SubjectOutcome(True, registry['source'], registry,
            {'classification': 'host-verified', 'speedup': 1.0, 'metrics': {'score': 100},
             'host_facts': [{'raw': 'x' * 2000}] * 500}, store.root)
    proposal = {'summary': 'Transfer tail safety', 'preconditions': 'Logical extent is known',
                'before': {'sha256': 'a' * 64, 'start_line': 1, 'end_line': 20000, 'text': 'before\n' * 20000},
                'after': {'sha256': 'b' * 64, 'start_line': 1, 'end_line': 20000, 'text': 'after\n' * 20000}}
    proposal_bytes = canonical(proposal)
    proposal_sha = __import__('hashlib').sha256(proposal_bytes).hexdigest()
    commands = []
    for name in ('control', 'treatment'):
        for role in ('log:actor', 'result:execution'):
            commands += [{'list_artifacts': {'prefix': name + ':' + role}},
                         {'read_artifact': {'sha256': registries[name][role]}}]
    hidden = registries['control']['source:prior-0499']
    commands += [{'list_artifacts': {'prefix': 'control:source:prior-', 'offset': 499, 'limit': 20}},
                 {'read_artifact': {'sha256': hidden}}, {'read_artifact': {'sha256': proposal_sha}}]
    evidence = [registries[name][role] for name in ('control', 'treatment')
                for role in ('log:actor', 'result:execution')] + [hidden, proposal_sha]
    replies = iter([*commands, dict(decision='accept', reason='Read actual matched logs and full-proposal page.',
                                   evidence=evidence, instructions=None)])
    contexts = []
    def complete(messages, *, remaining_seconds):
        assert len(json.dumps(messages, ensure_ascii=False).encode()) <= 40000
        context = json.loads(messages[1]['content'])
        contexts.append(context)
        assert context['proposal_sha256'] == proposal_sha
        for name in arms:
            arm = context['arms'][name]
            assert len(arm['artifacts']) <= 16 and arm['artifact_count'] == len(registries[name])
            assert arm['observations']['metrics'] == {'score': 100}
            assert arm['observations']['speedup'] == 1.0
            assert 'host_facts' not in arm['observations']
            assert registries[name]['log:actor'] not in arm['artifacts'].values()
        return json.dumps(next(replies))
    agent = MemoryVerifierAgent(complete)
    result = agent.review(proposal, arms, lambda digest, **kw: store.read_bytes(digest), max_seconds=60)
    assert result['evidence'] == evidence
    pages = [event['observation'] for event in agent.last_trace if event['kind'] == 'artifact-list']
    assert pages[-1] == {'artifacts': {'control:source:prior-0499': hidden}, 'total': 500, 'next_offset': None}
    read = {event['observation']['sha256']: event['observation'] for event in agent.last_trace if event['kind'] == 'read'}
    assert read[hidden]['text'] == store.read_bytes(hidden).decode()
    assert read[proposal_sha]['text'] == proposal_bytes[:8192].decode()
    assert read[proposal_sha]['next_offset'] == 8192
