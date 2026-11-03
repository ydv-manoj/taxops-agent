"""Stage 2 reward for AgentTune GRPO.

    reward = (field_accuracy + success) * efficiency
    efficiency = 1 / (1 + 0.1 * wasted_calls + 0.05 * calls_over_minimum)

- No submission scores 0, and any submission with something right scores above 0,
  so waste shrinks the reward but quitting never beats trying.
- Calculator calls don't count toward "over the minimum" (using one is fine), but
  a repeated calculator call still counts as wasted.
"""

from __future__ import annotations

import json
from typing import Any

from taxops.metrics import EpisodeScore, score_episode

WASTE_WEIGHT = 0.1
OVER_MIN_WEIGHT = 0.05


def episode_reward(score: EpisodeScore, calls: list[dict]) -> float:
    if not score.submitted:
        return 0.0
    non_calc = sum(c.get("name") != "calculator" for c in calls)
    over_min = max(0, non_calc - score.min_calls)
    efficiency = 1.0 / (1.0 + WASTE_WEIGHT * score.wasted_calls + OVER_MIN_WEIGHT * over_min)
    return (score.field_accuracy + float(score.success)) * efficiency


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def calls_from_agenttune(trajectory: Any) -> list[dict]:
    """Turn an AgentTune rollout Trajectory (object or dict) into our call list.

    Each step's metadata has tool_calls (OpenAI-style or bare) and tool_results as
    (name, result) pairs. AgentTune runs a step's calls in parallel and puts unknown
    tools first, so results are matched to calls by name, in order.
    """
    calls: list[dict] = []
    for step in _get(trajectory, "steps", []) or []:
        meta = _get(step, "metadata", {}) or {}
        results = [list(r) for r in meta.get("tool_results", []) or []]
        for tc in meta.get("tool_calls", []) or []:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or "unknown"
            args = fn.get("arguments", tc.get("arguments", tc.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None  # malformed arguments -> counted as an invalid call
            match = next((r for r in results if r[0] == name), None)
            result = ""
            if match is not None:
                results.remove(match)
                result = match[1]
            if isinstance(result, dict) and "error" in result:
                result = f"ERROR: {result['error']}"
            calls.append({"name": name, "arguments": args, "result": str(result)})
    return calls


def taxops_reward(prompts=None, completions=None, trajectories=None, case_id=None, **kwargs) -> list[float]:
    """AgentTune/TRL reward function. Needs a `case_id` column in the train dataset."""
    n = len(completions or trajectories or [])
    ids = case_id if isinstance(case_id, list) else [case_id] * n
    trajs = trajectories or [None] * n
    out = []
    for cid, traj in zip(ids, trajs, strict=False):
        if cid is None or traj is None:
            out.append(0.0)
            continue
        calls = calls_from_agenttune(traj)
        out.append(episode_reward(score_episode(cid, calls), calls))
    return out
