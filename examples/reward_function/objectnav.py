"""Reward function for object navigation GRPO training.

Scores model responses on three axes:
  - Format (weight 0.2): Response has <think>...</think> + <summary>...</summary> + action tag
  - Action validity (weight 0.3): Action tag is parseable with valid coordinates
  - Accuracy (weight 0.5): Scored using environment metadata when available (online mode),
    or expert action comparison (offline/SFT mode)

Supports two ground_truth formats:

**Offline (SFT-derived):**
  {"action_type": str, "expert_action": str,
   "answer_coordinates": [x,y]|null, "expert_coordinates": [x,y]|null}

**Online (env rollout):**
  {"distance_to_target_before": float, "distance_to_target_after": float,
   "target_visible": bool, "target_2d_coords": [x,y]|null,
   "target_position": {"x":..,"y":..,"z":..}, "executed_action_type": str,
   "step_idx": int, "episode_success": bool, ...}
"""

import json
import math
import re
from typing import Any

REWARD_NAME = "objectnav"
REWARD_TYPE = "batch"

# ── Parsing helpers ─────────────────────────────────────────────────────


def _check_format(response: str) -> float:
    """Check if response matches <think>...</think>\\n<summary>...</summary>\\n<action>."""
    pattern = re.compile(
        r"<think>.*?</think>\s*<summary>.*?</summary>\s*<(?:explore|answer).*?(?:/>|</(?:explore|answer)>)",
        re.DOTALL,
    )
    return 1.0 if pattern.search(response) else 0.0


def _parse_action(response: str) -> dict | None:
    """Parse the action tag from a response. Returns None if unparseable."""
    # <answer>(x,y)</answer>
    m = re.search(r"<answer>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</answer>", response)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return {"type": "answer", "coords": [x, y]}
        return None

    # <explore>ground:(x,y)</explore>
    m = re.search(r"<explore>\s*ground:\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</explore>", response)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return {"type": "explore_ground", "coords": [x, y]}
        return None

    # <explore>direction:ANGLE</explore>
    m = re.search(r"<explore>\s*direction:\s*(-?\d+)\s*</explore>", response)
    if m:
        angle = int(m.group(1))
        if -180 <= angle <= 180:
            return {"type": "explore_direction", "angle": angle}
        return None

    return None


def _coord_distance(a: list[int], b: list[int]) -> float:
    """Euclidean distance between two [x,y] coords on [0,1000] grid."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ── Reward computation ──────────────────────────────────────────────────

FORMAT_WEIGHT = 0.2
VALIDITY_WEIGHT = 0.3
ACCURACY_WEIGHT = 0.5

# Distance threshold for coordinate proximity (on [0,1000] scale)
CLOSE_THRESHOLD = 100.0  # ~10% of image


def _is_trajectory_metadata(gt: dict) -> bool:
    """Check if the ground truth contains pre-computed trajectory reward."""
    return "trajectory_reward" in gt


def _is_online_metadata(gt: dict) -> bool:
    """Check if the ground truth contains online env metadata."""
    return "distance_to_target_before" in gt


def _accuracy_score_online(parsed_action: dict | None, gt: dict) -> float:
    """Score accuracy using online environment metadata (distance-based)."""
    if parsed_action is None:
        return 0.0

    # Answer action: reward based on proximity to target's 2D projection
    if parsed_action["type"] == "answer":
        target_2d = gt.get("target_2d_coords")
        if target_2d is not None:
            dist = _coord_distance(parsed_action["coords"], target_2d)
            # High reward for accurate pointing at visible target
            return max(0.1, 1.0 / (1.0 + (dist / CLOSE_THRESHOLD) ** 2))
        else:
            # Answering when target not visible — penalize
            return 0.05

    # Explore actions: reward based on distance improvement toward target
    dist_before = gt.get("distance_to_target_before")
    dist_after = gt.get("distance_to_target_after")

    if dist_before is not None and dist_after is not None:
        # Distance improvement ratio
        if dist_before > 0.1:
            improvement = (dist_before - dist_after) / dist_before
            # Scale: -1.0 (moved away a lot) to 1.0 (reached target)
            # Map to [0.1, 1.0] reward range
            score = 0.5 + 0.5 * max(-0.8, min(1.0, improvement))
            return max(0.1, score)
        else:
            # Already very close — any action is okay
            return 0.8

    # Fallback: at least the action was valid
    return 0.3


def _accuracy_score_offline(parsed_action: dict | None, gt: dict) -> float:
    """Score accuracy using offline expert action comparison (SFT-derived data)."""
    if parsed_action is None:
        return 0.0

    expert_type = gt.get("action_type")

    # Type match bonus
    type_matches = False
    if expert_type == "answer" and parsed_action["type"] == "answer":
        type_matches = True
    elif expert_type == "explore_ground" and parsed_action["type"] == "explore_ground":
        type_matches = True
    elif expert_type == "explore_direction" and parsed_action["type"] == "explore_direction":
        type_matches = True

    if not type_matches:
        # Wrong action type — small partial credit if it's at least a valid action
        return 0.1

    # For answer actions: score based on coordinate proximity to expert
    if parsed_action["type"] == "answer":
        expert_coords = gt.get("answer_coordinates") or gt.get("expert_coordinates")
        if expert_coords:
            dist = _coord_distance(parsed_action["coords"], expert_coords)
            # Smooth reward: 1.0 at dist=0, 0.5 at CLOSE_THRESHOLD, drops to ~0.1 at 3x threshold
            return max(0.1, 1.0 / (1.0 + (dist / CLOSE_THRESHOLD) ** 2))
        return 0.5  # type matches but no ground truth coords

    # For explore_ground: coordinate proximity to expert
    if parsed_action["type"] == "explore_ground":
        expert_coords = gt.get("expert_coordinates")
        if expert_coords:
            dist = _coord_distance(parsed_action["coords"], expert_coords)
            return max(0.2, 1.0 / (1.0 + (dist / CLOSE_THRESHOLD) ** 2))
        return 0.5

    # For explore_direction: type match is enough (angle matching is noisy)
    if parsed_action["type"] == "explore_direction":
        return 0.7

    return 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute reward scores for a batch of model responses.

    Args:
        reward_inputs: Each dict has:
            - response: model output text
            - ground_truth: JSON string with env metadata (online or offline format)

    Returns:
        List of score dicts with keys: overall, format, validity, accuracy
    """
    scores = []
    for ri in reward_inputs:
        response = ri.get("response", "")
        gt_str = ri.get("ground_truth", "{}")
        try:
            gt = json.loads(gt_str)
        except (json.JSONDecodeError, TypeError):
            gt = {}

        fmt = _check_format(response)
        parsed = _parse_action(response)
        validity = 1.0 if parsed is not None else 0.0

        # Choose accuracy scorer based on metadata format
        if _is_trajectory_metadata(gt):
            # Trajectory-level reward already computed by the env rollout.
            # Return it directly as the overall score (format/validity still
            # provide useful metrics).
            scores.append({
                "overall": gt["trajectory_reward"],
                "format": fmt,
                "validity": validity,
                "accuracy": gt["trajectory_reward"],
            })
            continue
        elif _is_online_metadata(gt):
            accuracy = _accuracy_score_online(parsed, gt)
        else:
            accuracy = _accuracy_score_offline(parsed, gt)

        overall = FORMAT_WEIGHT * fmt + VALIDITY_WEIGHT * validity + ACCURACY_WEIGHT * accuracy

        scores.append({
            "overall": overall,
            "format": fmt,
            "validity": validity,
            "accuracy": accuracy,
        })

    return scores


# ── Self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # Perfect answer (offline)
        {
            "response": "<think>\nI see the laptop on the desk.\n</think>\n<summary>Found laptop.</summary>\n<answer>(320,410)</answer>",
            "ground_truth": json.dumps({"action_type": "answer", "expert_action": "<answer>(320,410)</answer>", "answer_coordinates": [320, 410]}),
        },
        # Correct format, wrong coordinates (offline)
        {
            "response": "<think>\nLooking around.\n</think>\n<summary>Exploring.</summary>\n<answer>(800,900)</answer>",
            "ground_truth": json.dumps({"action_type": "answer", "expert_action": "<answer>(320,410)</answer>", "answer_coordinates": [320, 410]}),
        },
        # Correct explore (offline)
        {
            "response": "<think>\nMoving forward.\n</think>\n<summary>Walking to doorway.</summary>\n<explore>ground:(450,720)</explore>",
            "ground_truth": json.dumps({"action_type": "explore_ground", "expert_action": "<explore>ground:(450,720)</explore>", "expert_coordinates": [450, 720]}),
        },
        # Bad format
        {
            "response": "I'll go left",
            "ground_truth": json.dumps({"action_type": "explore_direction"}),
        },
        # Online: good distance improvement
        {
            "response": "<think>\nMoving toward kitchen.\n</think>\n<summary>Walking to kitchen.</summary>\n<explore>ground:(500,600)</explore>",
            "ground_truth": json.dumps({
                "distance_to_target_before": 5.0,
                "distance_to_target_after": 3.0,
                "target_visible": False,
                "target_2d_coords": None,
                "executed_action_type": "explore_ground",
                "step_idx": 2,
                "episode_success": False,
            }),
        },
        # Online: answer with visible target
        {
            "response": "<think>\nI see the target!\n</think>\n<summary>Found it.</summary>\n<answer>(400,350)</answer>",
            "ground_truth": json.dumps({
                "distance_to_target_before": 1.5,
                "distance_to_target_after": 1.5,
                "target_visible": True,
                "target_2d_coords": [420, 360],
                "executed_action_type": "answer",
                "step_idx": 5,
                "episode_success": True,
            }),
        },
        # Online: answer when target not visible
        {
            "response": "<think>\nI think it's here.\n</think>\n<summary>Guessing.</summary>\n<answer>(200,300)</answer>",
            "ground_truth": json.dumps({
                "distance_to_target_before": 8.0,
                "distance_to_target_after": 8.0,
                "target_visible": False,
                "target_2d_coords": None,
                "executed_action_type": "answer",
                "step_idx": 1,
                "episode_success": False,
            }),
        },
    ]

    results = compute_score(test_cases)
    for tc, r in zip(test_cases, results):
        print(f"Response: {tc['response'][:60]}...")
        print(f"  Scores: {r}")
        print()
