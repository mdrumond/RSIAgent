import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory
from rsi.learning import Learning, Validation
from rsi.roles import SkillEdit, SkillProposal
from rsi.skills import builtin_bundle


class Role:
    calls = 0
    proposal = None

    def skillwriter(self, prompt, evidence, *, deadline):
        self.calls += 1
        return self.proposal or SkillProposal('prevent cancellation', (evidence[0].evidence_id,),
            (SkillEdit('debug', 'replace', 'Compare against an independent numeric reference.'),))


@pytest.fixture
def env(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'cas')
    role = Role()
    bundle = store.put_json(builtin_bundle().to_dict())
    memory = SubjectMemory(tmp_path / 'memory')
    baseline = store.put_bytes(memory.snapshot())
    packets = {}
    def load(row):
        return packets[row['original_id']]
    suite = store.put_json({'host_suite': 'passed'})
    def make(mode='phase2', validate=None):
        validate = validate or (lambda *args: Validation(True, True, suite))
        return Learning(tmp_path / 'learning.json', store, memory, role,
                        load_project=load, validate=validate, mode=mode)
    def project(name='p', recovery=False):
        ref = store.put_json({'result': 'numeric failure', 'project': name})
        packets[name] = dict(project_id=name, outcome='correctness_failure',
            evidence={ref: {'kind': 'failure', 'summary': 'cancellation detected'}},
            queries=[], usage=[], raw_refs=[ref])
        return dict(original_id=name, artifact_ref=ref), dict(operation_id=name, recovery=recovery,
            remaining={'actor_seconds': 60})
    return store, role, bundle, memory, baseline, packets, make, project


def test_memory_failure_not_correctness_and_phase1_never_writes(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    learner = make('phase1')
    update = learner.memory_update(baseline, [row], ctx)
    record = memory.records()[0]
    assert record['proposal']['kind'] == 'failure'
    assert record['proposal']['evidence'] == [row['artifact_ref']]
    assert store.read_json(update.reference)['memories']
    assert learner.skill_update(bundle, update.reference, row, dict(ctx, operation_id='s')).reference == bundle
    assert role.calls == 0


@pytest.mark.parametrize('gate', [Validation(False, True, 'harmful'), Validation(None, None, 'unknown')])
def test_candidate_is_immutable_and_bad_gate_keeps_incumbent(env, gate):
    store, role, bundle, memory, baseline, packets, make, project = env
    gate = Validation(gate.workflow, gate.correctness, store.put_json({'result': gate.evidence_ref}))
    learner = make(validate=lambda *args: gate)
    row, ctx = project()
    update = learner.skill_update(bundle, baseline, row, ctx)
    assert update.reference != bundle
    assert learner.active == bundle
    assert store.read_json(bundle) == builtin_bundle().to_dict()
    assert learner.evaluate(update.reference, ('case',), 'v') == bundle
    assert learner.active == bundle


def test_promote_rollback_and_restart_no_duplicate(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    learner = make()
    candidate = learner.skill_update(bundle, baseline, row, ctx).reference
    assert make().skill_update(bundle, baseline, row, ctx).reference == candidate
    assert role.calls == 1
    assert learner.evaluate(candidate, ('case',), 'v') == candidate
    assert make().evaluate(candidate, ('case',), 'v') == candidate
    learner.rollback(bundle)
    assert learner.active == bundle


def test_unsupported_reference_rejected_and_uncertain_call_not_repeated(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    role.proposal = SkillProposal('claim', ('invented',), (SkillEdit('debug', 'replace', 'x'),))
    learner = make()
    with pytest.raises(ValueError):
        learner.skill_update(bundle, baseline, row, ctx)
    with pytest.raises(RuntimeError, match='reconcile'):
        make().skill_update(bundle, baseline, row, ctx)
    assert role.calls == 1


def test_pruning_counts_unique_originals_and_preserves_safety(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [{'skill': 'debug', 'section': 'learned-guidance', 'kind': 'exposed'}]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    row, ctx = project('0', True)
    learner.memory_update(baseline, [row], dict(ctx, operation_id='recovery'))
    assert learner.removal_eligible('debug', 'learned-guidance')
    assert not learner.removal_eligible('debug', 'contract', safety=True)
    assert len(learner.state['projects']) == 10


def test_negative_receipts_retained_without_frequency_claim(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    negative = {'negative_records': [{'reason': 'unstable sum'}], 'results': []}
    q = store.put_json({'kind': 'retrieval', 'payload': {'response': negative}})
    packets['p']['queries'] = [q]
    learner = make()
    learner.memory_update(baseline, [row], ctx)
    assert memory.records()[0]['provenance']['queries'] == [q]
    assert store.read_json(q)['payload']['response'] == negative


def test_remove_only_exact_fragment_after_unique_exposures(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    candidate = None
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [{'skill': 'debug', 'section': None, 'kind': 'exposed'}]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    row, ctx = project('last')
    ref = row['artifact_ref']
    packets['last']['evidence'][ref]['kind'] = 'review'
    original = builtin_bundle().document('debug').read('learned-guidance')
    role.proposal = SkillProposal('reviewed removal', (ref,),
                                 (SkillEdit('debug', 'remove', original.strip()),))
    candidate = learner.skill_update(bundle, baseline, row, dict(ctx, operation_id='s')).reference
    from rsi.skills import SkillBundle
    changed = SkillBundle.from_dict(store.read_json(candidate)).document('debug')
    assert not changed.read('learned-guidance').strip()
    assert changed.read('contract') == builtin_bundle().document('debug').read('contract')


def test_citation_is_not_observable_use_but_application_is(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [dict(skill='debug', section=None, kind=k)
                                  for k in ('exposed', 'read', 'cited')]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    assert learner.removal_eligible('debug', 'learned-guidance')
    row, ctx = project('used')
    packets['used']['usage'] = [dict(skill='debug', section=None, kind='applied')]
    learner.memory_update(baseline, [row], dict(ctx, operation_id='m-used'))
    assert not learner.removal_eligible('debug', 'learned-guidance')


def test_quarantine_blocks_even_passing_validation(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    packets['p']['evidence'][row['artifact_ref']]['kind'] = 'counterevidence'
    learner = make()
    candidate = learner.skill_update(bundle, baseline, row, ctx).reference
    learner.quarantine(candidate, row['artifact_ref'])
    assert learner.evaluate(candidate, ('case',), 'v') == bundle


def test_failed_role_requires_reconciliation_and_stop_blocks_new_work(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    def fail(*args, **kwargs):
        role.calls += 1
        raise TimeoutError('unknown provider result')
    role.skillwriter = fail
    learner = make()
    with pytest.raises(TimeoutError):
        learner.skill_update(bundle, baseline, row, ctx)
    learner.stop()
    with pytest.raises(RuntimeError, match='reconcile'):
        make().skill_update(bundle, baseline, row, ctx)
    learner.reconcile(ctx['operation_id'], not_dispatched=True)
    with pytest.raises(RuntimeError, match='stopped'):
        learner.skill_update(bundle, baseline, row, ctx)
    assert role.calls == 1


def test_completed_proposal_reconciles_frozen_packet_without_provider_replay(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    learner = make()
    def lost_return(prompt, evidence, *, deadline):
        role.calls += 1
        role.saved = store.put_json({'rationale': 'repair', 'evidence_ids': [evidence[0].evidence_id],
            'edits': [{'skill': 'debug', 'operation': 'replace', 'content': 'Compare numeric error.'}]})
        raise ConnectionError('lost response envelope')
    role.skillwriter = lost_return
    with pytest.raises(ConnectionError):
        learner.skill_update(bundle, baseline, row, ctx)
    packets.clear()
    resumed = make()
    candidate = resumed.reconcile_proposal(ctx['operation_id'], role.saved, 2).reference
    assert resumed.active == bundle
    assert candidate != bundle
    assert role.calls == 1
