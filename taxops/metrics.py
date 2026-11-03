"""Score one episode: did the return come out right, and how much tool work was wasted?

An episode is the list of tool calls the agent made, in order, each as
{"name": str, "arguments": dict, "result": str}.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from taxops.env import RETURN_FIELDS, TOOLS, Case, _status, generate_case

TOOL_NAMES = {t.__name__ for t in TOOLS}


@dataclass
class EpisodeScore:
    case_id: str
    submitted: bool
    success: bool           # every field correct
    field_accuracy: float   # share of the 11 fields correct (0 if not submitted)
    tool_calls: int
    wasted_calls: int       # redundant + failed + invalid
    redundant_calls: int    # repeats, distractor reads, calls after submitting
    failed_calls: int       # the tool returned an ERROR
    invalid_calls: int      # unknown tool name or arguments that don't fit
    min_calls: int          # fewest non-calculator calls that can solve this case
    calculator_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def min_calls(case: Case) -> int:
    # list once, read each relevant document once, standard deduction, compute_tax, submit.
    return 1 + sum(d.relevant for d in case.documents) + 3


def _same(a, b) -> bool:
    if isinstance(b, str):
        return _status(a) == b
    try:
        return int(round(float(str(a).replace(",", "").replace("$", "")))) == b
    except ValueError:
        return False


def _key(call: dict) -> str:
    return call.get("name", "") + json.dumps(call.get("arguments") or {}, sort_keys=True, default=str)


def score_episode(case_id: str, calls: list[dict]) -> EpisodeScore:
    case = generate_case(case_id)
    seen: set[str] = set()
    redundant = failed = invalid = 0
    submission = None
    for call in calls:
        name, args = call.get("name"), call.get("arguments") or {}
        if name not in TOOL_NAMES or not isinstance(args, dict):
            invalid += 1
            continue
        if str(call.get("result", "")).startswith("ERROR"):
            failed += 1
            continue
        if submission is not None:
            redundant += 1  # anything after the first submit is wasted, including resubmits
            continue
        key = _key(call)
        if key in seen or (name == "list_documents" and any(k.startswith("list_documents") for k in seen)):
            redundant += 1
        elif name == "read_document" and (doc := case.doc(str(args.get("doc_id", "")).strip())) and not doc.relevant:
            redundant += 1
        seen.add(key)
        if name == "submit_return":
            submission = args

    correct = 0
    if submission is not None:
        correct = sum(f in submission and _same(submission[f], case.answer[f]) for f in RETURN_FIELDS)
    return EpisodeScore(
        case_id=case_id,
        submitted=submission is not None,
        success=correct == len(RETURN_FIELDS),
        field_accuracy=correct / len(RETURN_FIELDS),
        tool_calls=len(calls),
        wasted_calls=redundant + failed + invalid,
        redundant_calls=redundant,
        failed_calls=failed,
        invalid_calls=invalid,
        min_calls=min_calls(case),
        calculator_calls=sum(c.get("name") == "calculator" for c in calls),
    )


def summarize(scores: list[EpisodeScore]) -> dict:
    n = len(scores) or 1
    solved = [s for s in scores if s.success]
    return {
        "episodes": len(scores),
        "success_rate": sum(s.success for s in scores) / n,
        "field_accuracy": sum(s.field_accuracy for s in scores) / n,
        "submit_rate": sum(s.submitted for s in scores) / n,
        "avg_tool_calls": sum(s.tool_calls for s in scores) / n,
        "avg_wasted_calls": sum(s.wasted_calls for s in scores) / n,
        "failed_call_rate": sum(s.failed_calls for s in scores) / max(1, sum(s.tool_calls for s in scores)),
        # Efficiency only counts on solved cases, so "do nothing" can't look efficient.
        "avg_calculator_calls": sum(s.calculator_calls for s in scores) / n,
        # Calculator calls are excluded: using one is good practice, not waste.
        "calls_over_minimum_when_solved": (sum(s.tool_calls - s.calculator_calls - s.min_calls for s in solved)
                                           / len(solved))
        if solved else None,
    }
