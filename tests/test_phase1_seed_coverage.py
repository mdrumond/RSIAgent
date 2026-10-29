"""Copied memory contributes context without becoming new campaign history."""
import base64
from copy import deepcopy
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum import Capability, ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from test_a5_phase1_adaptive import _memory, _project, _result, _wave


def coordinator(tmp_path, initial, complete=lambda _: ''):
    return AdaptivePhase1Coordinator(tmp_path, 'Learn new kernels', initial,
        CurriculumAgent(complete), lambda req: _result(req.spec, True, 'verified'),
        max_parallel=8, project_target=64, max_projects=64)


def seed_bytes():
    memory = _memory(ProjectSpec.from_mapping(_project('seed-copy', 'gm-ub-movement')))
    return canonical_bytes({'schema': 'ascend-reviewed-memory-seed-v1',
        'snapshot_id': 'authenticated', 'phase1_memory': [memory.as_dict()]})


def test_seed_augments_future_context_only(tmp_path, monkeypatch):
    seed = seed_bytes()
    calls = []
    def authenticate(root, snapshot_id):
        calls.append((root, snapshot_id))
        return seed, root / 'snapshot'
    monkeypatch.setattr('benchmarks.ascend.reviewed_memory.reviewed_seed', authenticate)
    runner = coordinator(tmp_path, seed)
    state = runner._load_state()
    before = deepcopy(state)
    state_bytes = (tmp_path / 'state.json').read_bytes()
    snapshot = runner._memory_snapshot(state)
    context = runner._context(state, snapshot)
    assert context.pre_wave_coverage == (Capability.GM_UB_MOVEMENT,)
    assert [row.project_id for row in context.prior_outcomes] == ['seed-copy']
    assert context.prior_outcomes[0].verified
    assert 'lesson seed-copy' in context.prior_outcomes[0].summary
    assert context.curriculum_memory[0]['lesson_claims'] == ['lesson seed-copy']
    assert context.remaining_projects == 64
    assert runner._completed_attempts(state) == 0
    assert state == before
    assert (tmp_path / 'state.json').read_bytes() == state_bytes
    assert runner._memory_snapshot(state) == snapshot
    assert calls == [(tmp_path, 'authenticated')]

    # New learning and failed attempts remain new-only budget/history records.
    new = _memory(ProjectSpec.from_mapping(_project('new-add', 'simd-vector')))
    state['prior_outcomes'].append({'project_id': 'new-add', 'spec': new.project_spec.as_dict(),
        'verified': True, 'summary': 'new verified project'})
    state['memory_bytes_base64'].append(base64.b64encode(canonical_bytes(new.as_dict())).decode())
    state['coverage'].append('simd-vector')
    context = runner._context(state, snapshot)
    assert set(context.pre_wave_coverage) == {Capability.GM_UB_MOVEMENT, Capability.SIMD_VECTOR}
    assert len(context.prior_outcomes) == 2
    assert context.remaining_projects == 63
    assert runner._completed_attempts(state) == 1
    assert state['coverage'] == ['simd-vector']


@pytest.mark.parametrize('failure', ['different', 'missing'])
def test_seed_authentication_is_required(tmp_path, monkeypatch, failure):
    def authenticate(*_):
        if failure == 'missing':
            raise ValueError('reviewed snapshot is missing or ambiguous')
        return b'{}', tmp_path
    monkeypatch.setattr('benchmarks.ascend.reviewed_memory.reviewed_seed', authenticate)
    runner = coordinator(tmp_path, seed_bytes())
    state = runner._load_state()
    with pytest.raises(ValueError, match='reviewed'):
        runner._context(state, runner._memory_snapshot(state))


def test_legacy_initial_memory_does_not_authenticate(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmarks.ascend.reviewed_memory.reviewed_seed',
        lambda *_: pytest.fail('legacy memory is not a portable seed'))
    runner = coordinator(tmp_path, b'ordinary old memory')
    state = runner._load_state()
    assert runner._context(state, runner._memory_snapshot(state)).pre_wave_coverage == ()


def test_rejected_dependencies_are_corrected_and_audited_on_exhaustion(tmp_path):
    calls = []
    raw = []
    now = [0.0]
    def complete(messages):
        calls.append(messages)
        now[0] += 1
        response = json.loads(_wave(messages[1]['content'],
            [_project('new-add', 'simd-vector')]).split('\n', 1)[1])
        response['projects'][0]['requires'] = ['cube-tiling']
        value = 'DECISION: WAVE\n' + json.dumps(response)
        raw.append(value)
        return value
    runner = coordinator(tmp_path, b'legacy', complete)
    runner.clock = runner.curriculum.clock = lambda: now[0]
    runner.max_seconds = 3
    with pytest.raises(ValueError, match='rejected selection artifact:') as error:
        runner.propose_wave()
    assert len(calls) == 3
    assert "project new-add: unsupported requires ['cube-tiling']" in str(error.value)
    assert 'allowed pre-wave coverage prerequisites: []' in str(error.value)
    digest = str(error.value).rsplit(': ', 1)[1]
    artifact = ContentAddressedArtifacts(tmp_path / 'artifacts').read_json(digest)
    assert [row['content'] for row in artifact['messages'] if row['role'] == 'assistant'] == raw
    assert len([row for row in artifact['messages'] if row['role'] == 'user']) == 4
    assert not (tmp_path / 'pending.json').exists()


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('requires', [[], ['gm-ub-movement']])
def test_unfamiliar_study_operation_is_not_a_prerequisite(native, requires):
    from benchmarks.ascend.targets import a3_curriculum_policy
    from test_a5_phase1_curriculum_selector import context, wave
    spec = _project('study-new-add', 'simd-vector')
    ctx = context(wave_index=2, pre_wave_coverage=(Capability.GM_UB_MOVEMENT,))
    raw = json.loads(wave([spec], coverage=['gm-ub-movement'], wave_id='wave-002').split('\n', 1)[1])
    raw['projects'][0]['requires'] = requires
    policy = a3_curriculum_policy() if native else None
    decision, _ = CurriculumAgent(lambda _: 'DECISION: WAVE\n' + json.dumps(raw), target_policy=policy).select(ctx)
    assert decision.wave.projects[0].requires == tuple(Capability(row) for row in requires)
    assert Capability.SIMD_VECTOR not in ctx.pre_wave_coverage


def test_sibling_requirement_correction_preserves_exact_allowed_set():
    from test_a5_phase1_curriculum_selector import context, wave
    first = _project('producer', 'simd-vector')
    second = _project('consumer', 'gm-ub-movement')
    bad = json.loads(wave([first, second]).split('\n', 1)[1])
    bad['projects'][1]['requires'] = ['simd-vector']
    replies = ['DECISION: WAVE\n' + json.dumps(bad), wave([first, second])]
    calls = []
    def complete(messages):
        calls.append(messages)
        return replies[len(calls)-1]
    decision, _ = CurriculumAgent(complete).select(context())
    assert len(decision.wave.projects) == 2
    feedback = calls[1][-1]['content']
    assert 'project consumer: unsupported requires' in feedback
    assert 'allowed pre-wave coverage prerequisites: []' in feedback
