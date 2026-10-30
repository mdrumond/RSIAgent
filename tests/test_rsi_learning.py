import json
from hashlib import sha256
import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory
from rsi.learning import Learning, Validation
from rsi.roles import SkillEdit, SkillProposal
from rsi.skills import builtin_bundle


class Role:
    calls = 0
    proposal = None
    receipt = {'role': 'skillwriter', 'success': True, 'elapsed_seconds': 2.0}

    def skillwriter(self, prompt, evidence, *, deadline, wall_deadline=None):
        self.calls += 1
        context = json.loads(prompt[prompt.index('{"aggregate"'):])
        return self.proposal or SkillProposal('prevent cancellation', (evidence[0].evidence_id,),
            (SkillEdit('debug', 'replace', context['skills']['debug'] + 'Compare against an independent numeric reference.'),))


@pytest.fixture
def env(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'cas')
    role = Role()
    bundle = store.put_json(builtin_bundle().to_dict())
    memory = SubjectMemory(tmp_path / 'memory')
    baseline = store.put_bytes(memory.snapshot())
    packets = {}
    def load(row):
        packet = packets[row['original_id']]
        for event in packet['usage']:
            doc = builtin_bundle().document(event['skill'])
            event.setdefault('skill_hash', doc.digest)
            event.setdefault('content_hash', sha256(doc.read(event.get('section')).encode()).hexdigest())
        return packet
    suite = store.put_json({'host_suite': 'passed'})
    def make(mode='phase2', validate=None, **kwargs):
        validate = validate or (lambda *args: Validation(True, True, suite))
        return Learning(tmp_path / 'learning.json', store, memory, role,
                        load_project=load, validate=validate, mode=mode,
                        role_elapsed=lambda op: role.receipt['elapsed_seconds'] if role.receipt else None, **kwargs)
    def project(name='p', recovery=False):
        ref = store.put_json({'result': 'numeric failure', 'project': name})
        packets[name] = dict(project_id=name, outcome='correctness_failure',
            evidence={ref: {'kind': 'failure', 'summary': 'cancellation detected'}},
            queries=[], usage=[], raw_refs=[ref], usage_complete=True)
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
    assert learner.removal_eligible('debug', 'learned-guidance', document=builtin_bundle().document('debug'))
    assert not learner.removal_eligible('debug', 'contract', safety=True, document=builtin_bundle().document('debug'))
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


def test_negative_avoidance_citation_and_application_both_protect_guidance(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [dict(skill='debug', section=None, kind=k)
                                  for k in ('exposed', 'read')]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    assert learner.removal_eligible('debug', 'learned-guidance', document=builtin_bundle().document('debug'))
    row, ctx = project('used')
    packets['used']['usage'] = [dict(skill='debug', section=None, kind='cited', evidence_ref=row['artifact_ref'])]
    learner.memory_update(baseline, [row], dict(ctx, operation_id='m-used'))
    assert not learner.removal_eligible('debug', 'learned-guidance', document=builtin_bundle().document('debug'))


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
    def lost_return(prompt, evidence, *, deadline, wall_deadline=None):
        role.calls += 1
        role.saved = store.put_json({'rationale': 'repair', 'evidence_ids': [evidence[0].evidence_id],
            'edits': [{'skill': 'debug', 'operation': 'replace', 'content': builtin_bundle().document('debug').read('learned-guidance') + 'Compare numeric error.'}]})
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


def custom_bundle(store, content):
    from rsi.skills import SkillBundle, SkillDocument
    original = builtin_bundle()
    return store.put_json(SkillBundle(tuple(SkillDocument('debug', content) if d.name == 'debug' else d
                                          for d in original.documents), original.mapping).to_dict())


def test_yaml_metadata_and_missing_learned_slot_preserve_baseline(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    content = '---\ndescription: >-\n  Numeric debugging\n  with references\nname: "debug"\n---\n# User baseline\n\n## Checks\nCompare independent results.\n'
    incumbent = custom_bundle(store, content)
    row, ctx = project()
    learner = make()
    candidate = learner.skill_update(incumbent, baseline, row, ctx).reference
    changed = learner._bundle(candidate).document('debug')
    assert changed.read('checks') == learner._bundle(incumbent).document('debug').read('checks')
    assert changed.read('learned-guidance').strip() == 'Compare against an independent numeric reference.'
    assert store.read_json(incumbent)['documents'] != store.read_json(candidate)['documents']


@pytest.mark.parametrize('operation', ['replace', 'remove'])
@pytest.mark.parametrize('reviewed', [False, True])
def test_baseline_correction_requires_evidence_and_host_validation(env, operation, reviewed):
    store, role, bundle, memory, baseline, packets, make, project = env
    content = '---\nname: debug\ndescription: "Debugging"\n---\n# User baseline\n\n## Checks\nIgnore numeric errors.\n'
    incumbent = custom_bundle(store, content)
    row, ctx = project()
    if reviewed:
        packets['p']['evidence'][row['artifact_ref']]['kind'] = 'counterevidence'
    replacement = content.replace('Ignore numeric errors.', 'Check independent references.')
    role.proposal = SkillProposal('correct baseline', (row['artifact_ref'],),
        (SkillEdit('debug', operation, replacement if operation == 'replace' else 'Ignore numeric errors.'),))
    learner = make()
    if not reviewed:
        with pytest.raises(ValueError, match='requires host review'):
            learner.skill_update(incumbent, baseline, row, ctx)
    else:
        candidate = learner.skill_update(incumbent, baseline, row, ctx).reference
        assert learner.active == incumbent
        changed = learner._bundle(candidate).document('debug')
        assert changed.read('checks') != learner._bundle(incumbent).document('debug').read('checks')
        assert learner.evaluate(candidate, ('numeric-reference-case',), 'v') == candidate


def test_setup_excluded_and_unlimited_parent_uses_technical_request_timeout(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    now = [0.0]
    deadlines = []
    def thinking(prompt, evidence, *, deadline, wall_deadline=None):
        deadlines.append(deadline)
        now[0] += 102  # 100 seconds framework setup, two seconds thinking.
        return SkillProposal('repair', (evidence[0].evidence_id,), (SkillEdit('debug', 'replace', builtin_bundle().document('debug').read('learned-guidance') + 'Check error.'),))
    role.skillwriter = thinking
    row, ctx = project()
    ctx['remaining']['actor_seconds'] = None
    learner = make(clock=lambda: now[0])
    update = learner.skill_update(bundle, baseline, row, ctx)
    assert deadlines == [600]
    assert now[0] == 102
    assert update.actor_seconds == 2
    assert learner.state['operations'][ctx['operation_id']]['role_finished']['elapsed_seconds'] == 2


def test_missing_role_timing_suspends_instead_of_charging_wall(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    role.receipt = None
    row, ctx = project()
    learner = make()
    with pytest.raises(RuntimeError, match='accounting incomplete'):
        learner.skill_update(bundle, baseline, row, ctx)
    assert learner.state['operations'][ctx['operation_id']]['accounting_incomplete']
    assert not learner.state['candidates']
    with pytest.raises(RuntimeError, match='reconcile'):
        make().skill_update(bundle, baseline, row, ctx)
    assert role.calls == 1


def test_performance_evidence_ranks_after_error_recovery(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    perf = store.put_json({'seconds': 1.0})
    packets['p']['evidence'][perf] = {'kind': 'performance', 'summary': 'faster'}
    learner = make()
    learner.memory_update(baseline, [row], ctx)
    assert [item['kind'] for item in learner.aggregate()['outcomes']] == ['failure', 'performance']


@pytest.mark.parametrize('reviewed', [False, True])
def test_section_replace_cannot_bypass_removal_checks(env, reviewed):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    if reviewed:
        packets['p']['evidence'][row['artifact_ref']]['kind'] = 'review'
    role.proposal = SkillProposal('rewrite', (row['artifact_ref'],),
                                 (SkillEdit('debug', 'replace', 'Replacement guidance.'),))
    learner = make()
    if reviewed:
        candidate = learner.skill_update(bundle, baseline, row, ctx).reference
        assert learner._bundle(candidate).document('debug').read('learned-guidance').strip() == 'Replacement guidance.'
        assert learner.active == bundle
    else:
        with pytest.raises(ValueError, match='deletes guidance'):
            learner.skill_update(bundle, baseline, row, ctx)


def test_historical_evidence_is_admitted_and_restart_scoped(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    old, ctx = project('old')
    learner.memory_update(baseline, [old], dict(ctx, operation_id='memory-old'))
    row, ctx = project('new')
    role.proposal = SkillProposal('historical failure prevention', (old['artifact_ref'],),
        (SkillEdit('debug', 'replace', builtin_bundle().document('debug').read('learned-guidance') + 'Retest cancellation.'),))
    candidate = learner.skill_update(bundle, baseline, row, ctx).reference
    admitted = store.read_json(learner.state['candidates'][candidate]['admitted'])
    assert old['artifact_ref'] in admitted
    packets.clear()
    assert make().skill_update(bundle, baseline, row, ctx).reference == candidate
    assert role.calls == 1


def test_version_scoped_pruning_and_incomplete_telemetry(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    old = builtin_bundle().document('debug')
    from rsi.skills import SkillDocument
    changed = SkillDocument('debug', old.content + 'New error recovery guidance.\n')
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [dict(skill='debug', section=None, kind='exposed')]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    assert learner.removal_eligible('debug', 'learned-guidance', document=old)
    assert not learner.removal_eligible('debug', 'learned-guidance', document=changed)
    # New-version exposures do not inherit old-version citations.
    for i in range(10, 20):
        row, ctx = project(str(i))
        packets[str(i)]['usage_complete'] = False
        packets[str(i)]['usage'] = [dict(skill='debug', section=None, kind='exposed', skill_hash=changed.digest)]
        learner.memory_update(baseline, [row], dict(ctx, operation_id='m' + str(i)))
    assert not learner.removal_eligible('debug', 'learned-guidance', document=changed)


def test_old_citation_does_not_protect_changed_guidance(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    learner = make()
    old = builtin_bundle().document('debug')
    from rsi.skills import SkillDocument
    changed = SkillDocument('debug', old.content + 'New guidance.\n')
    row, ctx = project('old-used')
    packets['old-used']['usage'] = [dict(skill='debug', section=None, kind='cited')]
    learner.memory_update(baseline, [row], ctx)
    for i in range(10):
        row, ctx = project(str(i))
        packets[str(i)]['usage'] = [dict(skill='debug', section=None, kind='exposed',
            skill_hash=changed.digest, content_hash=changed.digest)]
        learner.memory_update(baseline, [row], ctx)
    assert learner.removal_eligible('debug', 'learned-guidance', document=changed)
    assert not learner.removal_eligible('debug', 'learned-guidance', document=old)


def test_duplicate_candidate_preserves_quarantine_and_validation_decisions(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project()
    packets['p']['evidence'][row['artifact_ref']]['kind'] = 'counterevidence'
    learner = make()
    candidate = learner.skill_update(bundle, baseline, row, ctx).reference
    learner.evaluate(candidate, ('case',), 'v')
    first_validation = dict(learner.state['candidates'][candidate]['validation'])
    learner.quarantine(candidate, row['artifact_ref'])
    again = learner.skill_update(bundle, baseline, row, dict(ctx, operation_id='again')).reference
    assert again == candidate
    record = learner.state['candidates'][candidate]
    assert record['quarantined'] == row['artifact_ref']
    assert record['validation'] == first_validation
    assert len(record['proposals']) == 2
    assert len(record['validations']) == 1
    assert learner.evaluate(candidate, ('case',), 'v2') == bundle
    assert len(record['validations']) == 2
    with pytest.raises(ValueError, match='quarantined'):
        learner.rollback(candidate)


def test_later_regression_can_quarantine_and_survives_restart(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project('original')
    learner = make()
    candidate = learner.skill_update(bundle, baseline, row, ctx).reference
    assert learner.evaluate(candidate, ('case',), 'v') == candidate
    later, later_ctx = project('regression')
    counter = later['artifact_ref']
    packets['regression']['candidate_ref'] = candidate
    with pytest.raises(ValueError, match='counterevidence'):
        learner.quarantine(candidate, counter, evidence_row=later)
    packets['regression']['evidence'][counter]['kind'] = 'counterevidence'
    learner.quarantine(candidate, counter, evidence_row=later)
    resumed = make()
    assert resumed.active == bundle
    assert resumed.state['candidates'][candidate]['quarantined'] == counter
    admitted = resumed.state['candidates'][candidate]['quarantine_evidence'][-1]
    assert store.read_json(admitted['packet'])['project_id'] == 'regression'
    with pytest.raises(ValueError, match='quarantined'):
        resumed.rollback(candidate)


def test_writer_corrects_frozen_full_baseline_without_retrieval_tools(env, tmp_path):
    store, role, bundle, memory, baseline, packets, make, project = env
    from rsi.skills import SkillDocument
    original = '---\nname: debug\ndescription: Numeric debugging\n---\n## Numeric policy\nIgnore numeric errors.\n'
    source = tmp_path / 'SKILL.md'
    source.write_text(original)
    incumbent = custom_bundle(store, source.read_text())
    source.write_text('changed after snapshot')
    row, ctx = project()
    packets['p']['evidence'][row['artifact_ref']]['kind'] = 'review'
    packets['p']['correction_skills'] = ['debug']
    observed = []
    def correct(prompt, evidence, *, deadline, wall_deadline=None):
        role.calls += 1
        context = json.loads(prompt[prompt.index('{"aggregate"'):])
        supplied = context['baseline_documents']['debug']
        frozen = SkillDocument('debug', supplied['content'])
        assert supplied['bundle_ref'] == incumbent
        assert frozen.digest == supplied['skill_hash']
        observed.append(frozen.digest)
        replacement = frozen.content.replace(frozen.read('numeric-policy'), 'Check independent references.\n')
        return SkillProposal('reviewed numerical correction', (row['artifact_ref'],),
                             (SkillEdit('debug', 'replace', replacement),))
    role.skillwriter = correct
    learner = make()
    candidate = learner.skill_update(incumbent, baseline, row, ctx).reference
    assert observed == [SkillDocument('debug', original).digest]
    assert learner._bundle(candidate).document('debug').read('numeric-policy') == 'Check independent references.\n'
    assert learner.active == incumbent
    frozen_context = store.read_json(learner.state['operations'][ctx['operation_id']]['writer_context'])
    assert frozen_context['baseline_documents']['debug']['content'] == original
    assert learner.evaluate(candidate, ('numeric-case',), 'v') == candidate


def test_observations_reach_subject_memory_search_and_writer_as_labelled_evidence(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    row, ctx = project('numeric')
    observations = dict(goal='accurate compensated summation', final_source='frozen-source-hash',
        diagnostics=[dict(command_ref=row['artifact_ref'], result='cancellation reproduced')],
        strategies=[dict(status='model_claim', strategy='compensated summation may recover accuracy')])
    packets['numeric']['observations'] = observations
    learner = make('phase1')
    learner.memory_update(baseline, [row], ctx)
    hits = memory.search('compensated')
    assert len(hits) == 1
    assert hits[0]['proposal']['observations'] == observations
    assert hits[0]['proposal']['kind'] == 'failure'
    aggregate = learner.aggregate()
    observed = aggregate['observations'][0]
    assert observed['observations']['strategies'][0]['status'] == 'model_claim'
    assert store.read_json(observed['packet_ref'])['observations'] == observations
    assert observed['evidence_ids'] == [row['artifact_ref']]
    # A fresh Phase2 controller consumes the same trusted observations in role context.
    phase2_path = learner.path.with_name('phase2.json')
    from rsi.learning import Learning
    seen = []
    original_writer = role.skillwriter
    def inspect_context(prompt, evidence, *, deadline, wall_deadline=None):
        context = json.loads(prompt[prompt.index('{"aggregate"'):])
        seen.append(context['aggregate']['observations'][0]['observations'])
        return original_writer(prompt, evidence, deadline=deadline, wall_deadline=wall_deadline)
    role.skillwriter = inspect_context
    phase2 = Learning(phase2_path, store, memory, role, load_project=lambda r: packets[r['original_id']],
        validate=lambda *args: Validation(None, None, row['artifact_ref']), role_elapsed=lambda op: 2)
    phase2.skill_update(bundle, baseline, row, dict(ctx, operation_id='skills'))
    assert seen == [observations]
    assert phase2.active == bundle


def test_parent_wall_reaches_role_and_failure_receipt_is_consumed_once(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    now = [100.0]
    seen = []
    def timeout(prompt, evidence, *, deadline, wall_deadline):
        seen.append((deadline, wall_deadline))
        now[0] += 5  # setup 3, actual role work 2; host role_finished measured 2.
        raise TimeoutError('wall expired')
    role.skillwriter = timeout
    learner = make(clock=lambda: now[0])
    row, ctx = project()
    ctx['remaining']['wall_seconds'] = 5
    receipts = []
    learner.role_elapsed = lambda identity: receipts.append(identity) or 2.0
    with pytest.raises(TimeoutError):
        learner.skill_update(bundle, baseline, row, ctx)
    assert seen == [(160, 105)]
    assert learner.state['operations'][ctx['operation_id']]['role_finished']['elapsed_seconds'] == 2
    assert not learner.state['candidates']
    with pytest.raises(RuntimeError, match='reconcile'):
        learner.skill_update(bundle, baseline, row, ctx)
    assert receipts == [ctx['operation_id']]


def test_parent_wall_exhausted_in_preparation_never_dispatches_role(env):
    store, role, bundle, memory, baseline, packets, make, project = env
    now = [0.0]
    learner = make(clock=lambda: now[0])
    row, ctx = project()
    ctx['remaining']['wall_seconds'] = 1
    loader = learner.load_project
    def prepare(row):
        now[0] += 2
        return loader(row)
    learner.load_project = prepare
    with pytest.raises(TimeoutError, match='before role dispatch'):
        learner.skill_update(bundle, baseline, row, ctx)
    assert role.calls == 0
    assert learner.state['operations'][ctx['operation_id']]['not_dispatched']
