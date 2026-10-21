import json
import pytest
from benchmarks.ascend.phase2_time import Phase2TimeBudget


def test_time_budget_preserves_failed_work_and_resume_without_count_limits(tmp_path):
    now = [0.]
    clock = lambda: now[0]
    budget = Phase2TimeBudget(tmp_path, 20, 5, clock=clock)
    with pytest.raises(OSError):
        with budget.active(practice=True):
            now[0] += 3
            raise OSError('retained operation')
    resumed = Phase2TimeBudget(tmp_path, 20, 5, clock=clock)
    assert resumed.remaining == 17
    assert resumed.practice_remaining == 2
    for _ in range(10):
        with resumed.active():
            now[0] += .1
    assert resumed.remaining == pytest.approx(16)
    resumed.reset_practice()
    assert resumed.practice_remaining == 5


def test_crashed_reservation_is_charged_and_changed_budget_rejected(tmp_path):
    budget = Phase2TimeBudget(tmp_path, 20, clock=lambda: 0)
    budget.state['inflight_seconds'] = 20
    budget.save()
    assert Phase2TimeBudget(tmp_path, 20).remaining == 0
    with pytest.raises(ValueError, match='budgets changed'):
        Phase2TimeBudget(tmp_path, 21)
    assert 'inflight_seconds' not in json.loads(budget.path.read_bytes())


@pytest.mark.parametrize('maximum', [0, -1, True, float('nan'), float('inf')])
def test_invalid_time_budget(maximum, tmp_path):
    with pytest.raises(ValueError, match='finite positive seconds'):
        Phase2TimeBudget(tmp_path, maximum)


def test_observing_live_allocation_does_not_consume_it(tmp_path):
    now = [0.]
    budget = Phase2TimeBudget(tmp_path, 20, 5, clock=lambda: now[0])
    with budget.active(practice=True):
        now[0] = 3
        observed = Phase2TimeBudget(tmp_path, 20, 5, clock=lambda: now[0])
        assert observed.state['elapsed_seconds'] == 0
        assert budget.practice_remaining == 2
        with pytest.raises(RuntimeError, match='already active'):
            with observed.active():
                pass
    assert Phase2TimeBudget(tmp_path, 20, 5).remaining == 17


def test_practice_crash_reserves_only_admitted_practice_window(tmp_path):
    budget = Phase2TimeBudget(tmp_path, 20, 5, clock=lambda: 0)
    with budget.active():
        with budget.practice_work():
            retained = json.loads(budget.path.read_bytes())
            assert retained['inflight_seconds'] == 5
            retained.pop('owner_pid')
            crash = tmp_path / 'crash'
            crash.mkdir()
            (crash / 'time-budget.json').write_text(json.dumps(retained))
            recovered = Phase2TimeBudget(crash, 20, 5, clock=lambda: 0)
            assert recovered.remaining == 15
            assert recovered.practice_remaining == 0
        assert budget.state['inflight_seconds'] == 20
