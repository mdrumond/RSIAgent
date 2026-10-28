import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.focused_memory import FocusedMemoryVerifier, paired_measurements
from benchmarks.ascend.memory_admission import MemoryAdmission
from benchmarks.ascend.subject_execution import SubjectOutcome
from benchmarks.ascend.subject_memory import SubjectMemory
from optimizer.jobs import canonical, sha
from test_a5_phase1_adaptive import _project
from test_subject_campaign import proposal


class Harness:
    def __init__(self, root, *, fail=None, before_passes=False, after_passes=True, transfer_passes=True,
                 performance=False, measure=False, same_task=False):
        self.store = ContentAddressedArtifacts(root / 'shared')
        self.before = self.store.put_bytes(b'bad retained full source\n')
        self.after = self.store.put_bytes(b'fixed retained full source\n')
        self.proposal = dict(proposal(self.after), before=dict(sha256=self.before, start_line=1, end_line=1),
            before_unavailable_reason=None)
        self.request = dict(project_id='source', source_sha256=self.after, host_verified=True,
            artifacts={'contract': self.store.put_json(_project('source', 'gm-ub-movement'))})
        self.calls, self.prompts, self.now = [], [], 0
        self.fail, self.before_passes, self.after_passes, self.transfer_passes = fail, before_passes, after_passes, transfer_passes
        self.performance, self.same_task = performance, same_task
        self.verifier = FocusedMemoryVerifier(self.complete, self.select, self.replay, self.transfer,
            self.store.read_bytes, model_identity={'model': 'fixture'}, runtime_identity={'device': '0'},
            measure_pair=self.measure if measure else None, max_seconds=100, arm_seconds=20,
            clock=lambda: self.now, wall_clock=lambda: self.now)

    def select(self, proposal, **kw):
        assert self.calls == []
        return dict(project_id='related', spec=_project('related', 'gm-ub-movement' if self.same_task else 'padded-tail'))

    def outcome(self, source, directory, passed):
        self.now += 1
        name = Path(directory).name
        self.calls.append(name)
        if self.fail == name:
            raise ConnectionError('retained handle unresolved')
        store = ContentAddressedArtifacts(directory / 'artifacts')
        store.put_bytes(self.store.read_bytes(source))
        refs = dict(source=source, **{'result:host': store.put_json(dict(passed=passed)),
            'log:actor': store.put_json(dict(knowledge_lookup=name == 'transfer', applied=name == 'transfer'))})
        return SubjectOutcome(passed, source, refs,
            dict(classification='host-verified' if passed else 'host-mismatch'), store.root)

    def replay(self, request, source, directory, **kw):
        assert request == self.request
        return self.outcome(source, directory, self.before_passes if source == self.before else self.after_passes)

    def transfer(self, task, snapshot, directory, **kw):
        assert snapshot == b'treatment' or json.loads(snapshot)['memories']
        return self.outcome(self.after, directory, self.transfer_passes)

    def measure(self, request, before, after, directory, **kw):
        self.calls.append('measure-pair')
        return dict(runtime={'device': '0'}, contract_sha256=sha(canonical(request)),
            sources=dict(control=before, treatment=after), host_verified=True,
            pairs=[dict(control_us=10, treatment_us=8)]*3)

    def complete(self, messages, **kw):
        self.prompts.append(messages)
        context = json.loads(messages[1]['content'])
        read = {json.loads(row['content'])['sha256'] for row in messages[2:]
                if row['role'] == 'user' and 'sha256' in json.loads(row['content'])}
        required = {context['proposal_sha256']}
        for arm in context['arms'].values():
            required.update(arm['artifacts'].values())
        for digest in sorted(required-read):
            return json.dumps({'read_artifact': {'sha256': digest}})
        feedback = [json.loads(row['content']).get('error', '') for row in messages[2:] if row['role'] == 'user']
        blocked = any('cannot support' in text or 'does not reproduce' in text or 'performance acceptance' in text for text in feedback)
        return json.dumps(dict(decision='rewrite' if blocked else 'accept', reason='Specific retained repair applied in related KDB transfer.',
            evidence=sorted(read), instructions='Narrow the claim to demonstrated evidence.' if blocked else None,
            claim_kind='performance' if self.performance else 'repair'))

    def run(self, root, **kw):
        return self.verifier.evaluate(self.proposal, b'baseline', b'treatment', root,
            source_request=self.request, remaining_seconds=90, **kw)


def test_replays_retained_sources_then_single_transfer_and_reuses_terminal_result(tmp_path):
    h = Harness(tmp_path)
    result = h.run(tmp_path / 'evaluation')
    assert result['decision'] == 'accept' and result['claim_kind'] == 'repair'
    assert h.calls == ['control', 'treatment', 'transfer']
    assert result['evidence_stages']['control'] == dict(status='completed', host_verified=False, elapsed_seconds=1)
    assert all(result['validity'].values())
    assert result == h.run(tmp_path / 'evaluation', task=result['trial_task'])
    assert h.calls == ['control', 'treatment', 'transfer']
    h.request['project_id'] = 'changed'
    with pytest.raises(ValueError, match='identity changed'):
        h.run(tmp_path / 'evaluation')


@pytest.mark.parametrize('failure', ['control', 'treatment', 'transfer'])
def test_interrupted_stages_hold_without_duplicate_execution_or_judgment(tmp_path, failure):
    h = Harness(tmp_path, fail=failure)
    first = h.run(tmp_path / 'evaluation')
    calls = h.calls.copy()
    assert first['decision'] == 'inconclusive' and not h.prompts
    assert first['evidence_stages'][failure]['status'] == 'interrupted'
    assert first['assessment']['status'] == 'infrastructure-blocked'
    assert h.run(tmp_path / 'evaluation') == first and h.calls == calls


@pytest.mark.parametrize('change', ['missing-before', 'same-source', 'same-task', 'wrong-after'])
def test_missing_or_non_discriminating_evidence_never_dispatches_device_work(tmp_path, change):
    h = Harness(tmp_path, same_task=change == 'same-task')
    if change == 'missing-before':
        h.proposal['before'] = None
    elif change == 'same-source':
        h.proposal['before']['sha256'] = h.after
    elif change == 'wrong-after':
        h.proposal['after']['sha256'] = h.before
    result = h.run(tmp_path / 'evaluation')
    assert result['decision'] == 'inconclusive' and h.calls == []


@pytest.mark.parametrize('options', [dict(before_passes=True), dict(transfer_passes=False), dict(performance=True)])
def test_host_guards_reject_unsupported_judge_acceptance(tmp_path, options):
    h = Harness(tmp_path, **options)
    result = h.run(tmp_path / 'evaluation')
    assert result['decision'] == 'rewrite'
    assert result['instructions'] and len(h.prompts) > 1


def test_failed_repair_does_not_spend_a_fresh_actor(tmp_path):
    h = Harness(tmp_path, after_passes=False)
    assert h.run(tmp_path / 'evaluation')['decision'] == 'inconclusive'
    assert h.calls == ['control', 'treatment'] and h.prompts == []


def test_performance_needs_matched_repeated_host_correctness_evidence(tmp_path):
    h = Harness(tmp_path, before_passes=True, performance=True, measure=True)
    result = h.run(tmp_path / 'evaluation')
    assert result['decision'] == 'accept' and result['claim_kind'] == 'performance'
    assert result['evidence_stages']['measurement']['status'] == 'completed'
    report = h.measure(h.request, h.before, h.after, tmp_path)
    for change in [dict(pairs=report['pairs'][:1]), dict(host_verified=False), dict(runtime={'device': '1'}),
                   dict(contract_sha256='a'*64), dict(sources=dict(control=h.after, treatment=h.before))]:
        with pytest.raises(ValueError):
            paired_measurements(dict(report, **change), {'device': '0'}, h.request, h.before, h.after)


def test_normal_admission_publishes_only_after_focused_evidence_and_keeps_provenance(tmp_path):
    h = Harness(tmp_path)
    memory = SubjectMemory(tmp_path / 'memory', b'')
    gate = MemoryAdmission(tmp_path / 'gate', memory, h.verifier, lambda *a, **kw: pytest.fail('unexpected rewrite'))
    try:
        gate.submit('lesson', h.proposal, {}, h.request, remaining_seconds=90)
        gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome'] == 'accepted'
        verification = memory.records()[0]['provenance']['memory_verification']
        assert verification['mode'] == 'focused' and verification['claim_kind'] == 'repair'
        assert verification['evidence_stages']['transfer']['host_verified'] is True
    finally:
        gate.close()


def test_runtime_route_replays_full_source_against_retained_contract(tmp_path):
    from types import SimpleNamespace
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_a5_knowledge import FakeEmbeddings
    from test_subject_runtime import bound
    database = tmp_path / 'knowledge.sqlite'
    embeddings = FakeEmbeddings()
    reference = tmp_path / 'reference.txt'
    reference.write_text('Copy preserves logical values and padding.\n')
    with KnowledgeDB(database, embeddings) as db:
        db.index(tmp_path, [reference], collection='reference', language='ascend-c')
    selected, calls = [], []
    config = bound(tmp_path, kdb=database, embeddings=embeddings)
    config['coordinator'].curriculum.target_policy = SimpleNamespace(admit=selected.append)
    def execute(spec, source):
        calls.append((spec, source))
        return SimpleNamespace(artifacts={'observed': {'passed': True}},
            evidence=SimpleNamespace(verified=True, classification=SimpleNamespace(value='host-verified'), detail='oracle matches'))
    config['actor'].executor.execute = execute
    campaign = build_subject_campaign(dict(memory_admission_mode='focused', actor_memory_delivery='kdb-only'),
        target='a3', root=tmp_path / 'runtime', subjects=[Subject('s', 'copies')], initial_memory=b'',
        compose=lambda *a, **kw: config)
    try:
        assert isinstance(campaign.memory_gate.verifier, FocusedMemoryVerifier)
        contract = _project('retained', 'gm-ub-movement')
        source = campaign.artifacts.put_bytes(b'complete retained source\n')
        request = dict(project_id='retained', artifacts={'contract': campaign.artifacts.put_json(contract)})
        result = campaign.replay_memory_source(request, source, tmp_path / 'replay', remaining_seconds=30)
        assert result.host_verified and calls[0][1] == 'complete retained source\n'
        assert calls[0][0].as_dict() == contract and selected == [calls[0][0]]
        assert json.loads(result.read_artifact(result.artifacts['result:replay']))['classification'] == 'host-verified'
    finally:
        campaign.memory_gate.close()


@pytest.mark.parametrize('metadata', [
    {'capabilities': ['gm-ub-movement', 'failure-repair'], 'failure_mode': 'compile'},
    {'capabilities': ['gm-ub-movement', 'failure-repair'], 'failure_mode': 'host-mismatch'},
    {'learning_focus': ['sync-correctness']},
])
def test_metadata_only_variation_cannot_turn_source_replay_into_transfer(tmp_path, metadata):
    h = Harness(tmp_path, same_task=True)
    original = _project('source', 'gm-ub-movement')
    original.update(metadata)
    h.request['artifacts']['contract'] = h.store.put_json(original)
    result = h.run(tmp_path / 'evaluation')
    assert result['decision'] == 'inconclusive' and h.calls == []
    assert 'transfer must change' in result['error']


def test_completed_measurement_survives_process_loss_before_judgment(tmp_path):
    class ProcessLost(BaseException):
        pass
    h = Harness(tmp_path, before_passes=True, performance=True, measure=True)
    complete = h.verifier.judge.complete
    def lose_process(*args, **kw):
        raise ProcessLost()
    h.verifier.judge.complete = lose_process
    directory = tmp_path / 'evaluation'
    with pytest.raises(ProcessLost):
        h.run(directory)
    saved = json.loads((directory / 'evaluation.json').read_bytes())
    assert saved['stages']['measurement']['status'] == 'completed' and 'decision' not in saved
    calls = h.calls.copy()
    h.verifier.judge.complete = complete
    result = h.run(directory)
    assert result['decision'] == 'accept' and result['claim_kind'] == 'performance'
    assert h.calls == calls and h.calls.count('measure-pair') == 1
    assert result['evidence_stages']['measurement']['status'] == 'completed'
