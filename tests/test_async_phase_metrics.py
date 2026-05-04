"""Unit tests for `_compute_async_phase_metrics` (multiturn_env.py).

Pure-Python aggregator — no GPU, no Ray, no vLLM.
"""

from __future__ import annotations

from verl.workers.rollout.multiturn_env import (
    Trajectory,
    _compute_async_phase_metrics,
)


def _mk_traj(start, end, steps):
    """`steps` is a list of (prompt_ms, gen_ms, env_ms) tuples."""
    t = Trajectory(pool=None, slot_id=0, episode_id="x")
    t.traj_start_ts = start
    t.traj_end_ts = end
    t.step_phase_times = [
        {"prompt_ms": p, "gen_ms": g, "env_ms": e} for (p, g, e) in steps
    ]
    return t


def test_empty_input_returns_empty_dict():
    assert _compute_async_phase_metrics([], total_wall_s=1.0) == {}


def test_no_phase_times_returns_empty():
    t = Trajectory(pool=None, slot_id=0, episode_id="x")  # no timestamps set
    assert _compute_async_phase_metrics([t], total_wall_s=1.0) == {}


def test_phase_aggregates_match_expected_values():
    t1 = _mk_traj(0.0, 1.0, [(10.0, 50.0, 5.0), (12.0, 60.0, 6.0)])
    t2 = _mk_traj(0.5, 1.5, [(8.0, 40.0, 7.0)])

    out = _compute_async_phase_metrics([t1, t2], total_wall_s=1.5)

    # Sums match what we constructed.
    assert out["rollout/phase_prompt_ms_sum"] == 30.0
    assert out["rollout/phase_gen_ms_sum"] == 150.0
    assert out["rollout/phase_env_ms_sum"] == 18.0
    # Averages: prompt (10+12+8)/3=10, gen (50+60+40)/3=50, env (5+6+7)/3=6.
    assert out["rollout/phase_prompt_ms_avg"] == 10.0
    assert out["rollout/phase_gen_ms_avg"] == 50.0
    assert out["rollout/phase_env_ms_avg"] == 6.0
    assert out["rollout/total_wall_ms"] == 1500.0


def test_concurrency_calculation():
    # Two trajectories with overlapping windows:
    #   t1: [0, 1]   t2: [0.5, 1.5]
    # Expected: peak=2, time-weighted avg = (1*0.5 + 2*0.5 + 1*0.5) / 1.5 ≈ 1.333
    t1 = _mk_traj(0.0, 1.0, [(10.0, 50.0, 5.0)])
    t2 = _mk_traj(0.5, 1.5, [(10.0, 50.0, 5.0)])
    out = _compute_async_phase_metrics([t1, t2], total_wall_s=1.5)
    assert out["rollout/concurrency_peak"] == 2.0
    assert abs(out["rollout/concurrency_avg"] - 4.0 / 3.0) < 1e-6


def test_concurrency_serial_trajectories():
    # Non-overlapping: peak should be 1.
    t1 = _mk_traj(0.0, 1.0, [(10.0, 50.0, 5.0)])
    t2 = _mk_traj(2.0, 3.0, [(10.0, 50.0, 5.0)])
    out = _compute_async_phase_metrics([t1, t2], total_wall_s=3.0)
    assert out["rollout/concurrency_peak"] == 1.0


if __name__ == "__main__":
    test_empty_input_returns_empty_dict()
    test_no_phase_times_returns_empty()
    test_phase_aggregates_match_expected_values()
    test_concurrency_calculation()
    test_concurrency_serial_trajectories()
    print("OK")
