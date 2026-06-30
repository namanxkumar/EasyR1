"""Pure helpers for the moving forced-depth curriculum (prefix_force_depth).

Kept dependency-free (no torch/ray/vllm) so the controller rule is unit-testable
on a CPU node, independent of the heavy rollout module that calls it.

The curriculum forces ``G`` expert steps from each prefixed group's branch and
leaves ``L`` trailing steps for the on-policy student (the answer step always
among them). ``L`` is a single GLOBAL controlled variable, nudged once per
training iteration toward a target prefixed-group success rate.
"""

from __future__ import annotations


def forced_g_from_plan(plan_len: int | None, leave_on_policy: float) -> int:
    """Forced expert-step count from the branch.

    ``plan_len`` = expert actions remaining from the branch INCLUDING the
    terminal answer (``SimulatorPool.compute_expert_plan_length``). With
    ``L = round(leave_on_policy) >= 1`` the answer step is always left on policy.
    Returns ``max(0, plan_len - round(L))``; ``0`` (plan_len None or shorter than
    L) collapses the group to the legacy single-hint-at-branch path.
    """
    if plan_len is None:
        return 0
    return max(0, int(plan_len) - int(round(float(leave_on_policy))))


def step_leave_on_policy(
    leave_on_policy: float,
    ema: float,
    ema_seen: bool,
    success_rate: float,
    *,
    beta: float,
    target: float,
    deadband: float,
    floor: float,
) -> tuple[float, float, bool]:
    """One controller update toward ``target`` prefixed-group success.

    EMA-smooth the observed ``success_rate`` (first observation set directly),
    then: ema above ``target + deadband`` => student is coping, leave MORE on
    policy (``L += 1``, force fewer steps); ema below ``target - deadband`` => too
    hard, force MORE (``L -= 1``, floored at ``floor``); within the deadband =>
    unchanged. Returns ``(new_L, new_ema, True)``.
    """
    new_ema = success_rate if not ema_seen else beta * ema + (1.0 - beta) * success_rate
    if new_ema > target + deadband:
        new_L = leave_on_policy + 1.0
    elif new_ema < target - deadband:
        new_L = max(float(floor), leave_on_policy - 1.0)
    else:
        new_L = leave_on_policy
    return new_L, new_ema, True
