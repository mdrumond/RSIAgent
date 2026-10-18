"""Target-independent observations retain the A5 compatibility surface."""

import pytest

from benchmarks.ascend.contracts import CompileObservation, RuntimeObservation
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation as A5CompileObservation,
    RuntimeObservation as A5RuntimeObservation,
)


def test_a5_observation_imports_are_shared_types():
    assert A5CompileObservation is CompileObservation
    assert A5RuntimeObservation is RuntimeObservation


@pytest.mark.parametrize("passed,artifact", [(True, None), (False, "object")])
def test_compile_requires_artifact_exactly_on_success(passed, artifact):
    with pytest.raises(ValueError):
        CompileObservation(passed, "a" * 64, "compiler report", artifact)


def test_runtime_requires_complete_host_output():
    with pytest.raises(ValueError):
        RuntimeObservation(True, "b" * 64, "synchronized", None)
    assert RuntimeObservation(True, "b" * 64, "synchronized", (1.0,)).output == (1.0,)
