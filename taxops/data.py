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


def recovery_plan(case_id: str, rng: random.Random) -> tuple[list[Turn], int]:
    """An oracle run with one realistic mistake, then the fix, so the model learns to recover.
    Kinds: a wrong doc id, or a filing status the tools don't accept.
    Returns the plan and the index of the mistaken call, whose turn is left out of the loss."""
    plan = oracle_plan(case_id)
    kind = rng.choice(["bad_doc", "bad_status"])
    if kind == "bad_doc":
        i = next(k for k, turn in enumerate(plan) if turn[0][0] == "read_document")
        n_docs = len(env.generate_case(case_id).documents)
        plan.insert(i, [("read_document", {"case_id": case_id, "doc_id": f"D{n_docs + rng.randint(1, 5)}"})])
    else:
        i = next(k for k, turn in enumerate(plan) if any(n == "lookup_standard_deduction" for n, _ in turn))
        plan.insert(i, [("lookup_standard_deduction", {"filing_status": "joint married"})])
    return plan, i


def sft_conversations(case_ids: list[str], recovery_share: float = 0.2, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    schemas = tool_schemas()
    rows = []
    for cid in case_ids:
        recovery = rng.random() < recovery_share
        plan, skip = recovery_plan(cid, rng) if recovery else (oracle_plan(cid), None)
        ep = _replay(cid, plan, schemas)
        assert ep["stop"] == "submitted", cid
        # skip_assistant_turns: 0-based assistant-turn indices to exclude from the SFT loss.
        rows.append({"case_id": cid, "kind": "recovery" if recovery else "oracle", "messages": ep["messages"],
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
