"""Dataset splits and the Stage 1 (SFT) conversations.

Splits are fixed id ranges, so train and eval can never overlap:
  train       tiers easy/medium/hard, ids 0..699          (2,100 cases)
  eval        tiers easy/medium/hard, ids 100000..100099  (300 cases)
  eval-shift  tier "shift", ids 200000..200149            (150 cases)

    python3 -m taxops.data export out/   # writes the split files and sft.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from taxops import env
from taxops.agent_loop import run_episode, tool_schemas
from taxops.scripted import oracle

SPLITS = {
    "train": [f"{t}-{i:05d}" for t in env.TIERS for i in range(0, 700)],
    "eval": [f"{t}-{i:05d}" for t in env.TIERS for i in range(100_000, 100_100)],
    "eval-shift": [f"shift-{i:05d}" for i in range(200_000, 200_150)],
}


Turn = list[tuple[str, dict]]  # the calls sent together in one assistant reply


def _replay(case_id: str, plan: list[Turn], schemas: list[dict]) -> dict:
    """Run the real agent loop with a scripted 'model' that emits the planned turns."""

    def generate(messages):
        i = sum(m["role"] == "assistant" for m in messages)
        return json.dumps([{"tool_name": name, "parameters": args} for name, args in plan[i]])

    return run_episode(case_id, generate, schemas)


def oracle_plan(case_id: str) -> list[Turn]:
    turns: dict[int, Turn] = {}
    for c in oracle(case_id):
        turns.setdefault(c["turn"], []).append((c["name"], c["arguments"]))
    return [turns[k] for k in sorted(turns)]


def recovery_plan(case_id: str, rng: random.Random, kind: str | None = None) -> tuple[list[Turn], int]:
    """An oracle run with one realistic mistake, then the fix, so the model learns to recover.
    Kinds: a wrong doc id, a filing status the tools don't accept, a decoy read alongside the
    real documents, or (v4) the superseded original 1099-INT read next to its CORRECTED copy.
    The last two are what SFT v3 did on most of its failures: the oracle demos never open those
    documents, so the model never saw what to do once it had. Here the later turns leave them out.
    Returns the plan and the index of the mistaken turn, which is left out of the loss."""
    plan = oracle_plan(case_id)
    case = env.generate_case(case_id)
    kind = kind or rng.choice(["bad_doc", "bad_status"])
    if kind == "bad_doc":
        i = next(k for k, turn in enumerate(plan) if turn[0][0] == "read_document")
        plan.insert(i, [("read_document", {"case_id": case_id, "doc_id": f"D{len(case.documents) + rng.randint(1, 5)}"})])
    elif kind == "bad_status":
        i = next(k for k, turn in enumerate(plan) if any(n == "lookup_standard_deduction" for n, _ in turn))
        plan.insert(i, [("lookup_standard_deduction", {"filing_status": "joint married"})])
    else:
        if kind == "read_superseded":
            extra = [d for d in case.documents if not d.relevant and d.title.startswith("Form 1099-INT")
                     and any(c.relevant and c.title == d.title + " (CORRECTED)" for c in case.documents)]
        else:  # read_decoy
            extra = [d for d in case.documents if not d.relevant]
        i = next(k for k, turn in enumerate(plan) if turn[0][0] == "read_document")
        for d in rng.sample(extra, 1 if kind == "read_decoy" else len(extra)):
            plan[i].insert(rng.randint(0, len(plan[i])), ("read_document", {"case_id": case_id, "doc_id": d.doc_id}))
        plan[i].sort(key=lambda c: int(c[1]["doc_id"][1:]))  # read in folder order, like the model does
    return plan, i


def sft_conversations(case_ids: list[str], recovery_share: float = 0.2, superseded_share: float = 0.6,
                      seed: int = 0) -> list[dict]:
    """superseded_share: of the train cases with a CORRECTED 1099-INT, how many become
    read_superseded recoveries (v4). The rest stay as clean oracle runs or ordinary recoveries."""
    rng = random.Random(seed)
    schemas = tool_schemas()
    rows = []
    for cid in case_ids:
        case = env.generate_case(cid)
        kind = None
        if "corrected_1099" in case.notes and rng.random() < superseded_share:
            kind = "read_superseded"
        elif rng.random() < recovery_share:
            kinds = ["bad_doc", "bad_status"] + (["read_decoy"] * 2 if any(not d.relevant for d in case.documents) else [])
            kind = rng.choice(kinds)
        plan, skip = recovery_plan(cid, rng, kind) if kind else (oracle_plan(cid), None)
        ep = _replay(cid, plan, schemas)
        assert ep["stop"] == "submitted", cid
        # skip_assistant_turns: 0-based assistant-turn indices to exclude from the SFT loss.
        rows.append({"case_id": cid, "kind": kind or "oracle", "messages": ep["messages"],
                     "skip_assistant_turns": [] if skip is None else [skip]})
    return rows


def export(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in SPLITS.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for cid in ids:
                case = env.generate_case(cid)
                f.write(json.dumps({"case_id": cid, "tier": case.tier, "prompt": env.task_prompt(cid),
                                    "answer": case.answer, "min_calls": 3 + 1 + sum(d.relevant for d in case.documents),
                                    "traps": case.notes}) + "\n")
    rows = sft_conversations(SPLITS["train"])
    with open(out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {', '.join(f'{k}={len(v)}' for k, v in SPLITS.items())}, sft={len(rows)} to {out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2] if len(sys.argv) > 2 else "out")
    else:
        print(__doc__)
