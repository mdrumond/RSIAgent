"""Input propagation and rejection at charter boundaries."""

import pytest

from explore.charter import (
    self_evolving_actor_memory_distillation_msg,
    self_evolving_curriculum_charter,
)


def test_curriculum_requires_an_exact_target_seed():
    with pytest.raises(ValueError, match="target_direction"):
        self_evolving_curriculum_charter("  ")


def test_failed_memory_never_claims_success_and_invalid_outcomes_fail_closed():
    with pytest.raises(ValueError, match="PASS or FAIL"):
        self_evolving_actor_memory_distillation_msg("YIELD", "report")


def test_conditioned_curriculum_requires_exploration_mode():
    with pytest.raises(ValueError, match="requires phase1_exploration"):
        self_evolving_curriculum_charter("public query", phase1_target_conditioned=True)
