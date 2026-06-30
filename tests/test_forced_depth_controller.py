"""Unit tests for the moving forced-depth controller (prefix_force_depth).

CPU-only: the pure rule lives in ``verl.workers.rollout.forced_depth`` precisely
so it can be tested without importing the heavy rollout module.
"""

import importlib.util
import os

# Load the dependency-free module DIRECTLY by path so the test runs on a CPU node
# without triggering verl/__init__'s heavy imports (torch/ray/vllm/codetiming).
_MOD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "verl", "workers", "rollout", "forced_depth.py"
)
_spec = importlib.util.spec_from_file_location("forced_depth", _MOD_PATH)
_fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fd)
forced_g_from_plan = _fd.forced_g_from_plan
step_leave_on_policy = _fd.step_leave_on_policy


def test_forced_g_basic():
    # L=1 (default floor): force all but the terminal answer step.
    assert forced_g_from_plan(5, 1) == 4
    # Larger L leaves more on policy.
    assert forced_g_from_plan(5, 3) == 2
    # L >= plan_len collapses to no forcing (legacy single-hint path).
    assert forced_g_from_plan(2, 3) == 0
    assert forced_g_from_plan(1, 1) == 0
    # None plan length (oracle unavailable) => no forcing.
    assert forced_g_from_plan(None, 1) == 0
    # Rounding of fractional L.
    assert forced_g_from_plan(6, 1.4) == 5  # round(1.4)=1
    assert forced_g_from_plan(6, 1.6) == 4  # round(1.6)=2


def test_controller_first_obs_sets_ema_directly():
    L, ema, seen = step_leave_on_policy(
        1.0, 0.0, False, 0.8,
        beta=0.5, target=0.5, deadband=0.05, floor=1.0,
    )
    assert seen is True
    assert ema == 0.8  # first observation set directly, no smoothing
    assert L == 2.0    # 0.8 > 0.5+0.05 => leave more on policy


def test_controller_success_above_target_raises_L():
    # ema already seen at 0.7; new obs 0.7 -> stays 0.7 > target -> L+1.
    L, ema, seen = step_leave_on_policy(
        2.0, 0.7, True, 0.7,
        beta=0.5, target=0.5, deadband=0.05, floor=1.0,
    )
    assert ema == 0.7
    assert L == 3.0


def test_controller_success_below_target_lowers_L_floored():
    # Drop success to 0.0 from a low ema -> L decreases but floored at 1.
    L, ema, seen = step_leave_on_policy(
        2.0, 0.1, True, 0.0,
        beta=0.5, target=0.5, deadband=0.05, floor=1.0,
    )
    assert ema == 0.05  # 0.5*0.1 + 0.5*0.0
    assert L == 1.0     # 0.05 < 0.45 => L-1 = 1 (== floor)
    # Already at floor: stays at floor, never below.
    L2, _, _ = step_leave_on_policy(
        1.0, 0.0, True, 0.0,
        beta=0.5, target=0.5, deadband=0.05, floor=1.0,
    )
    assert L2 == 1.0


def test_controller_within_deadband_holds_L():
    # ema lands inside [target-deadband, target+deadband] -> L unchanged.
    L, ema, seen = step_leave_on_policy(
        3.0, 0.5, True, 0.5,
        beta=0.5, target=0.5, deadband=0.05, floor=1.0,
    )
    assert abs(ema - 0.5) <= 0.05
    assert L == 3.0


def test_controller_converges_toward_setpoint():
    # Closed-loop sanity: success is a decreasing function of L (forcing fewer
    # steps -> harder -> lower success). The controller should settle near the
    # setpoint rather than diverge.
    target, deadband = 0.5, 0.05

    def success_for_L(L):
        # Monotonic decreasing in L; ~0.5 around L=3.
        return max(0.0, min(1.0, 1.0 - 0.16 * (L - 1.0)))

    L, ema, seen = 1.0, 0.0, False
    for _ in range(40):
        gu = success_for_L(L)
        L, ema, seen = step_leave_on_policy(
            L, ema, seen, gu,
            beta=0.5, target=target, deadband=deadband, floor=1.0,
        )
    # Settles in a small band around the L where success ~= target.
    assert 2.0 <= L <= 5.0
    assert abs(ema - target) < 0.25
