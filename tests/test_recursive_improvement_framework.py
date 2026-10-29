"""Contracts for the exploration and frozen-memory evaluation framework."""

from dataclasses import fields
import json
from pathlib import Path

import pytest
import yaml

from config.settings import Config, load
from explore.phase1_wave import parse_wave_handoff
import benchmarks.osworld.pipeline as protocol


REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "config/osworld/rsi.example.json"
CONTROL_CONFIG = REPO / "config/roles/target.yaml"
PHASE3_CONFIG = REPO / "config/osworld/baseline.yaml"
PHASE3_LOCK = REPO / "config/osworld/evaluation.lock.json"


@pytest.fixture(autouse=True)
def prepared_release_assets(tmp_path, monkeypatch):
    """Exercise real release validation using a local metadata fixture."""
    lock = json.loads(PHASE3_LOCK.read_text())
    assets = lock['task_assets']
    root = tmp_path / 'OSWorld-V2'
    folder = root / assets['local_dir']
    folder.mkdir(parents=True)
    marker = {'benchmark_release': lock['task_release'],
              **{key: assets[key] for key in ('repository', 'repo_type', 'tag', 'commit')}}
    (folder / assets['release_marker']).write_text(json.dumps(marker))
    monkeypatch.setattr(protocol, 'OSWORLD_ROOT', root)
    monkeypatch.setenv('OSWORLD_FILE_BASE_URL', '')
    monkeypatch.setenv('WEBSITE_HOST_SUFFIX', '')


def _spec(tmp_path: Path, *, held_out=None,
          distribution=None) -> Path:
    value = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    value["run_name"] = "unit_protocol"
    value["study_design"] = "held_out_generalization"
    value["phase1"]["distribution_file"] = "config/osworld_v2_training_distribution.md"
    value["phase3"]["held_out_tasks"] = held_out or []
    if distribution is not None:
        value["phase1"]["distribution_file"] = str(distribution)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_example_declares_target_conditioned_phases_and_budget_checkpoints():
    spec = protocol.load_protocol(EXAMPLE)

    assert spec["phase1"]["project_budget"] == 8
    assert spec["phase1"]["checkpoints"] == [0, 4, 8]
    assert spec["_resolved"]["held_out_tasks"] == ("task_080",)
    assert protocol.preflight(spec, "phase1")[
        "official_grader_available_in_phase"] is False
    assert protocol.preflight(spec, "phase3")[
        "official_grader_available_in_phase"] is True
    assert protocol.preflight(spec, "phase3")[
        "memory_mutable_in_phase"] is False
    assert protocol.preflight(spec, "phase3")[
        "curriculum_enabled_in_phase"] is False
    lock = protocol._protocol_lock(spec, EXAMPLE.read_bytes())
    assert lock["boundaries"] == {
        "official_evaluator_calls_phase1": 0,
        "official_evaluator_phase3_only_after_agent_handoff": True,
        "phase3_memory_writeback": False,
        "learning_after_phase3_begins": False,
    }
    assert "verifier_control_config" in lock["configs"]["phase1"]


def test_preflight_and_child_share_release_setup_without_mutating_parent(
        tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "RESULTS_ROOT", tmp_path / "results")
    spec = protocol.load_protocol(_spec(tmp_path))
    before = dict(protocol.os.environ)
    report = protocol.preflight(spec, "phase1")
    release = report["release_environment"]
    assert release["user_simulator"]["model"] == "openai/gpt-4o"
    assert release["task_assets"]["mode"] == "release_bound_local_snapshot"
    assert dict(protocol.os.environ) == before

    def launch(_argv, **kwargs):
        environment = kwargs["environment"]
        assert environment["OSWORLD_USER_SIM_MODEL"] == "openai/gpt-4o"
        assert environment["WEBSITE_HOST_SUFFIX"] == release["website_host_suffix"]
        assert environment["OSWORLD_FILE_BASE_URL"] == release["task_assets"]["path"]
        protocol._write_json_atomic(
            spec["_resolved"]["run_root"] / "phase1/result.json", {
                "status": "budget_exhausted", "official_evaluator_calls": 0})

    monkeypatch.setattr(protocol, "_run", launch)
    protocol.execute_phase1(spec)
    assert dict(protocol.os.environ) == before


def test_parallel_phase1_is_agent_authored_not_a_static_search_menu(tmp_path):
    value = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    value["run_name"] = "parallel_wave_unit"
    value["phase1"]["parallel_waves"] = True
    value["phase1"]["parallelism"] = 4
    path = tmp_path / "parallel.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = protocol.load_protocol(path)
    report = protocol.preflight(loaded, "phase1")
    lock = protocol._protocol_lock(loaded, path.read_bytes())

    assert report["phase1_parallel_waves"] is True
    assert report["phase1_parallelism"] == 4
    assert lock["phase1_parallel_waves"] is True
    assert lock["phase1_parallelism"] == 4


def test_parallel_wave_transport_accepts_free_topology_and_rejects_dependencies():
    raw = json.dumps({
        "decision": "WAVE",
        "rationale": "test two independent hypotheses",
        "projects": [
            {"id": "writer-format", "instruction":
             "Work in /home/user/evolution_project and save the deliverable there."},
            {"id": "mail-attach", "instruction":
             "Work in /home/user/evolution_project and keep all outputs there."},
        ],
    })
    decision = parse_wave_handoff(raw)
    assert decision is not None
    assert [project.project_id for project in decision.projects] == [
        "writer-format", "mail-attach"]
    assert parse_wave_handoff(json.dumps({
        "decision": "WAVE", "rationale": "invalid",
        "projects": [{"id": "bad", "instruction": "use sibling output"}],
    })) is None
    saturated = parse_wave_handoff(json.dumps({
        "decision": "SATURATED", "rationale": "marginal value is low",
        "projects": [],
    }))
    assert saturated is not None and saturated.projects == ()


def test_phase1_distribution_may_not_name_a_held_out_instance(tmp_path):
    distribution = tmp_path / "distribution.md"
    distribution.write_text(
        "Explore office workflows, especially task_019.", encoding="utf-8")
    path = _spec(
        tmp_path, held_out=["task_019"],
        distribution=distribution)
    with pytest.raises(protocol.ProtocolError, match="names held-out"):
        protocol.load_protocol(path)


def test_phase_profiles_keep_roles_agentic_and_phase_boundaries_distinct():
    known = {field.name for field in fields(Config)}
    for path in (CONTROL_CONFIG, PHASE3_CONFIG):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(raw) <= known

    practice = load(str(CONTROL_CONFIG))
    evaluation = load(str(PHASE3_CONFIG))
    for cfg in (practice, evaluation):
        assert cfg.model == "z-ai/glm-5.3"
        assert cfg.verifier_model == "moonshotai/kimi-k3"
        assert cfg.look_ensemble == 1
        assert cfg.verifier_continuity is True
        assert cfg.verifier_stage_lifecycle is False
        assert cfg.verifier_unverified_evidence is True
        assert cfg.verifier_execution_mode == "rollback_mirror"
        assert cfg.actor_evaluator_isolation is True
        assert cfg.env_memory_dir == ""
    assert practice.verifier_failure_starts_evolution is False
    assert practice.verifier_evolve_route is False
    assert evaluation.verifier_failure_starts_evolution is False
    assert evaluation.verifier_evolve_route is False

    lock = json.loads(PHASE3_LOCK.read_text(encoding="utf-8"))
    assert lock["actor_agent"]["sha256"] == protocol._sha256(PHASE3_CONFIG)
    assert lock["evaluator"]["visibility"] == \
        "sealed_after_actor_verifier_loop"
    assert lock["memory"]["host_writeback"] is False


@pytest.mark.parametrize('change', [lambda value: value.update(schema_version=1),
                                  lambda value: value.update(phase2={})])
def test_obsolete_protocol_is_rejected_before_outputs(tmp_path, monkeypatch, change):
    monkeypatch.setattr(protocol, 'RESULTS_ROOT', tmp_path / 'outputs')
    value = json.loads(EXAMPLE.read_text())
    change(value)
    path = tmp_path / 'old-protocol.json'
    path.write_text(json.dumps(value))
    with pytest.raises(protocol.ProtocolError):
        protocol.main(['--protocol', str(path), '--phase', 'phase1',
                       '--execute', 'RUN-RECURSIVE-IMPROVEMENT-PHASE1'])
    assert not (tmp_path / 'outputs').exists()


def phase1_handoff(tmp_path, monkeypatch):
    spec = protocol.load_protocol(_spec(tmp_path, held_out=['task_019']))
    monkeypatch.setattr(protocol, 'RSIAGENT_ROOT', tmp_path)
    root = spec['_resolved']['run_root'] = tmp_path / 'run'
    memory = root / 'phase1/memory_frozen'
    protocol._atomic_install_memory(str(memory), {'lesson.md': b'grounded lesson'})
    record = protocol._memory_record(memory)
    protocol._write_json_atomic(root / 'phase1/result.json', {
        'status': 'budget_exhausted', 'official_evaluator_calls': 0,
        'memory_frozen_path': str(memory), 'memory_tree_sha256': record['tree_sha256']})
    monkeypatch.setattr(protocol, '_configure_phase3_evaluator', lambda *_: {'sealed': True})
    monkeypatch.setattr(protocol, '_configure_release_environment', lambda *a, **k: {})
    return spec, memory


def test_phase3_consumes_exact_phase1_freeze_and_preserves_it(tmp_path, monkeypatch):
    spec, source = phase1_handoff(tmp_path, monkeypatch)
    launches = []
    def launch(argv, **kwargs):
        launches.append((argv, kwargs))
        frozen = Path(argv[argv.index('--memory-dir') + 1])
        assert protocol._read_memory_tree(str(frozen)) == {'lesson.md': b'grounded lesson'}
        tag = argv[argv.index('--tag') + 1]
        seed = argv[argv.index('--seed') + 1]
        protocol._write_json_atomic(tmp_path / 'results/task_019' / f'seed{seed}_{tag}/result.json', {
            'score': 0.75, 'status': 'done',
            'memory': {'tree_sha256': protocol._memory_record(frozen)['tree_sha256'], 'host_writeback': False},
            'official_evaluator_feedback_entered_agent': False})
    monkeypatch.setattr(protocol, '_run', launch)
    result = protocol.execute_phase3(spec)
    assert len(launches) == 1 and launches[0][1]['phase'] == 'phase3'
    assert result['mean_score'] == 0.75 and result['memory_updates'] == 0
    assert protocol._read_memory_tree(str(source)) == {'lesson.md': b'grounded lesson'}


def test_changed_phase1_memory_blocks_evaluation_before_writes(tmp_path, monkeypatch):
    spec, memory = phase1_handoff(tmp_path, monkeypatch)
    (memory / 'lesson.md').write_bytes(b'changed after publication')
    monkeypatch.setattr(protocol, '_run', lambda *_a, **_k: pytest.fail('must not launch'))
    with pytest.raises(protocol.ProtocolError, match='changed after publication'):
        protocol.execute_phase3(spec)
    assert not (spec['_resolved']['run_root'] / 'phase3').exists()


def test_evaluation_cannot_mutate_frozen_memory(tmp_path, monkeypatch):
    spec, _ = phase1_handoff(tmp_path, monkeypatch)
    def mutate(argv, **kwargs):
        frozen = Path(argv[argv.index('--memory-dir') + 1])
        (frozen / 'lesson.md').write_bytes(b'mutated')
        seed = argv[argv.index('--seed') + 1]
        tag = argv[argv.index('--tag') + 1]
        protocol._write_json_atomic(tmp_path / 'results/task_019' / f'seed{seed}_{tag}/result.json', {'score': 1})
    monkeypatch.setattr(protocol, '_run', mutate)
    with pytest.raises(protocol.ProtocolError, match='mutated frozen'):
        protocol.execute_phase3(spec)
    assert not (spec['_resolved']['run_root'] / 'phase3/result.json').exists()
