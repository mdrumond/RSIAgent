"""Connect ALE environments to the existing evaluator-free RSIAgent RSI loops."""

from __future__ import annotations

from pathlib import Path

from .practice import CleanPool, hooks_for
from .protocol import event, memory_record, role, write_json


def phase1(spec):
    from explore.phase1_wave import evolve_parallel_phase1
    from explore.practice_loop import _atomic_install_memory, _read_memory_tree

    root = Path(spec["work_dir"])
    pool = CleanPool(
        spec["practice_sandboxes"], practice=True, reset_endpoint=spec["reset_endpoint"]
    )
    try:
        hooks = hooks_for(pool, spec["corpus"], spec.get("audit_public_task"))
        result = evolve_parallel_phase1(
            root=str(root),
            target_direction=spec["instruction"],
            actor_cfg=role("actor"),
            curriculum_cfg=role("curriculum"),
            memory_cfg=role("memory"),
            verifier_control_cfg=role("target"),
            vm_factory=pool.acquire,
            corpus_path=spec["corpus"],
            hooks=hooks,
            event_sink=lambda kind, **payload: event(root, kind, payload),
            project_budget=8,
            checkpoint_projects=(0, 4, 8),
            max_parallel=4,
            target_query_conditioned=True,
            park_verified_branches=True,
        )
        _atomic_install_memory(
            str(root / "memory_frozen"), _read_memory_tree(result.memory_dir)
        )
        record = {
            "phase": "phase1",
            "status": result.status,
            "projects": result.projects,
            "reason": result.reason,
            "memory": memory_record(root / "memory_frozen"),
            "official_evaluator_calls": 0,
            "transition_ready": result.status in ("saturated", "budget_exhausted"),
        }
        write_json(root / "phase-result.json", record)
        return record
    finally:
        pool.close()
