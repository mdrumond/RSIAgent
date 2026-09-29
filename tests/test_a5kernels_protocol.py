from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.a5kernels.protocol import (
    ExecutionEnvironment,
    ExecutionPlan,
    SourceFile,
)


def _plan(**changes) -> ExecutionPlan:
    values = {
        "request_id": "request",
        "attempt_id": "attempt",
        "language": "catlass-dsl",
        "files": (SourceFile("kernel.py", "pass\n"),),
        "argv": ("python", "kernel.py"),
        "input_a": (1.0,),
        "input_b": (2.0,),
    }
    values.update(changes)
    return ExecutionPlan(**values)


def test_environment_is_canonical_and_renders_one_env_invocation():
    environment = ExecutionEnvironment(
        bindings=(("ZED", "2"), ("ALPHA", "1")),
        unset=("OLD_ZED", "OLD_ALPHA"),
    )

    assert environment.bindings == (("ALPHA", "1"), ("ZED", "2"))
    assert environment.unset == ("OLD_ALPHA", "OLD_ZED")
    assert environment.render(("python", "kernel.py")) == (
        "env", "-u", "OLD_ALPHA", "-u", "OLD_ZED",
        "ALPHA=1", "ZED=2", "python", "kernel.py",
    )


def test_environment_updates_are_immutable_and_move_between_policies():
    original = ExecutionEnvironment(bindings=(("KEEP", "1"),), unset=("MOVE",))
    bound = original.with_binding("MOVE", "2")
    unset = bound.with_unset("KEEP")

    assert original == ExecutionEnvironment(bindings=(("KEEP", "1"),), unset=("MOVE",))
    assert dict(bound.bindings) == {"KEEP": "1", "MOVE": "2"}
    assert bound.unset == ()
    assert unset.bindings == (("MOVE", "2"),)
    assert unset.unset == ("KEEP",)


@pytest.mark.parametrize(
    "environment, error",
    [
        (lambda: ExecutionEnvironment(bindings=(("BAD-NAME", "1"),)), "valid"),
        (lambda: ExecutionEnvironment(bindings=(("X", "1"), ("X", "2"))), "unique"),
        (lambda: ExecutionEnvironment(unset=("X", "X")), "unique"),
        (lambda: ExecutionEnvironment(bindings=(("X", "1"),), unset=("X",)), "both"),
    ],
)
def test_environment_rejects_invalid_or_ambiguous_policy(environment, error):
    with pytest.raises(ValueError, match=error):
        environment()


def test_environment_rejects_non_pair_binding_without_indexing_it():
    with pytest.raises(TypeError, match="string pairs"):
        ExecutionEnvironment(bindings=(("ONLY_NAME",),))


def test_environment_changes_execution_identity():
    plan = _plan()
    changed = replace(plan, environment=ExecutionEnvironment(bindings=(("DEVICE", "0"),)))

    assert plan.execution_id != changed.execution_id


@pytest.mark.parametrize("executable", ["env", "/usr/bin/env", "bin/env"])
def test_plan_rejects_env_wrapped_argv(executable):
    with pytest.raises(ValueError, match="env wrapper"):
        _plan(argv=(executable, "X=1", "python", "kernel.py"))
