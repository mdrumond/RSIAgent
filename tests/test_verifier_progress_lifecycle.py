"""A blocked Verifier preserves its VM and cannot enter official scoring."""
import json
from pathlib import Path
import runpy
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest

from config.settings import Config
from core.loop import LoopResult
from core.verifier_runtime import (
    AgenticVerifierInfrastructureError, AgenticVerifierNoProgressError,
)


@pytest.mark.parametrize("with_memory", [False, True])
@pytest.mark.parametrize("outcome", ["blocked", "blocked_disk_full", "infra", "complete"])
def test_task_runner_preserves_only_blocked_verification(monkeypatch, tmp_path, outcome, with_memory):
    repo = Path(__file__).resolve().parents[1]
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    monkeypatch.setenv("RSIAGENT_ROOT", str(repo))
    monkeypatch.setenv("OSWORLD_ROOT", str(benchmark))
    monkeypatch.delenv("RSIAGENT_CONFIG", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["benchmarks/osworld/task.py", "task_unit", "--seed", "17"])
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmarks.osworld.runtime._install_paths", lambda: None)

    public_instruction = f"Public task input for {tmp_path.name}"
    private_gold = f"private-gold-{tmp_path}"
    private_evaluator = f"private-evaluator-{tmp_path}"
    task_config = SimpleNamespace(instruction=public_instruction, user_config=None,
        evaluator={'gold': private_gold, 'function': private_evaluator})

    class Desktop:
        provider = SimpleNamespace(container=SimpleNamespace(id="retained-container"))
        controller = SimpleNamespace(http_server="http://retained-controller")
        user_simulator = None
        closed = False
        evaluations = 0
        candidate = b"exact candidate and live Verifier checkpoint"

        def reset(self, *, task_config):
            self.reset_config = task_config

        def close(self):
            self.closed = True
            self.candidate = None

        def evaluate(self):
            self.evaluations += 1
            return {"score": 0.25}

    desktop = Desktop()
    # No benchmark task code, Docker provider, guest command, or model is invoked.
    modules = {
        "benchmarks.osworld.provider": {
            "prepare_checkpointable_docker_provider": lambda _mode: None},
        "task_loader": {
            "resolve_task_json_path": lambda **_kwargs: "synthetic-public-task",
            "load_task_config": lambda *_args, **_kwargs: task_config},
        "desktop_env": {"__path__": []},
        "desktop_env.desktop_env": {"DesktopEnv": lambda **_kwargs: desktop},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    namespace = runpy.run_path(str(repo / "benchmarks/osworld/task.py"))
    main = namespace["main"]
    state = main.__globals__
    state["RSIAGENT_ROOT"] = tmp_path
    state["VM"] = lambda environment: environment
    memory = tmp_path / 'memory'
    memory.mkdir()
    memory_file = memory / f'lesson-{tmp_path.name}.txt'
    memory_file.write_bytes(b'allowed learned memory')
    cfg = Config(model="synthetic-actor", verifier_execution_mode="rollback_mirror",
        verifier_persist_scratch=True, actor_evaluator_isolation=False,
        env_memory_dir=str(memory) if with_memory else '')
    state["load_config"] = lambda _path: cfg
    pushes, preambles = [], []
    def push_memory(vm, path):
        pushes.append((vm, path))
        return True
    permitted_extra = f"Allowed memory inventory token {tmp_path}"
    def memory_preamble(inventory):
        preambles.append(inventory)
        return permitted_extra
    monkeypatch.setattr('explore.commit.push_memory', push_memory)
    monkeypatch.setattr('explore.charter.memory_preamble', memory_preamble)
    state["configure_associated_benchmark_lock"] = lambda *_args, **_kwargs: None
    state["provider_counts"] = lambda: {}
    state["reset_provider_counts"] = lambda: None
    calls = []

    def attempt(instruction, vm, config, sink, **kwargs):
        calls.append((instruction, config, kwargs))
        assert vm is desktop and vm.candidate is not None
        if outcome.startswith("blocked"):
            if outcome == "blocked_disk_full":
                def fail_write(_payload):
                    raise OSError("diagnostic disk full")
                sink.save_recovery = fail_write
            raise AgenticVerifierNoProgressError("3 consecutive segments without actions")
        if outcome == "infra":
            raise AgenticVerifierInfrastructureError("unrelated execution failure")
        return LoopResult(status="done"), []

    state["run_with_resume"] = attempt
    if outcome.startswith("blocked"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 75
    elif outcome == "infra":
        with pytest.raises(AgenticVerifierInfrastructureError):
            main()
    else:
        main()

    assert desktop.reset_config is task_config
    assert desktop.reset_config.evaluator == {'gold': private_gold, 'function': private_evaluator}
    assert calls == [(public_instruction, cfg, {
        'opening_extra': permitted_extra if with_memory else '', 'ask_user': None})]
    config_data = json.dumps(vars(calls[0][1]), default=str)
    assert private_gold not in config_data and private_evaluator not in config_data
    assert pushes == ([(desktop, str(memory))] if with_memory else [])
    assert preambles == ([f"  {memory_file.stat().st_size:>7}  {memory_file.name}"] if with_memory else [])
    root = tmp_path / "results/task_unit/seed17"
    if outcome.startswith("blocked"):
        assert desktop.closed is False
        assert desktop.candidate == b"exact candidate and live Verifier checkpoint"
        assert desktop.evaluations == 0
        assert not (root / "result.json").exists()
        assert not (root / "grading_checkpoint.json").exists()
        if outcome == "blocked":
            record = json.loads((root / "recovery.json").read_text())
            assert record["status"] == "blocked"
            assert record["vm_preserved"] is True
            assert record["container_id"] == "retained-container"
            assert record["official_evaluator_calls"] == 0
            assert record["recovery_requires_verifier_boundary_review"] is True
    else:
        assert desktop.closed is True
        assert desktop.evaluations == (1 if outcome == "complete" else 0)
