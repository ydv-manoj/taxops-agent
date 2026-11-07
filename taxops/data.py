"""Dataset splits and the Stage 1 (SFT) conversations.

Splits are fixed id ranges, so train and eval can never overlap:
  train       tiers easy/medium/hard, ids 0..699          (2,100 cases)
  eval        tiers easy/medium/hard, ids 100000..100099  (300 cases)
  eval-shift  tier "shift", ids 200000..200149            (150 cases)
  dev         tier "hard",  ids 110000..110029            (30 cases, for choosing between models)
  dev-shift   tier "shift", ids 300000..300049            (50 cases, same; never trained on)
  v5 training also uses extra train ids 700..99999 (see v5_conversations).

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
    "dev": [f"hard-{i:05d}" for i in range(110_000, 110_030)],
    "dev-shift": [f"shift-{i:05d}" for i in range(300_000, 300_050)],
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


def corrected_first(case_id: str, plan: list[Turn]) -> list[Turn]:
    """v6: read any CORRECTED form in a turn of its own, then the rest. The corrected form says
    "This form replaces the 1099-INT previously issued", so that text is in context when the
    model decides what else to open. v3 opened the superseded original in 19 of the 44 shift
    cases that had one and got every one of them wrong. Same calls, one more turn."""
    return _split_corrected(case_id, plan)[0]


def _split_corrected(case_id: str, plan: list[Turn]) -> tuple[list[Turn], int | None]:
    """corrected_first, plus the index of the turn that was split (None if nothing changed)."""
    case = env.generate_case(case_id)
    corrected = {d.doc_id for d in case.documents if d.title.endswith("(CORRECTED)")}
    is_corr = lambda c: c[0] == "read_document" and c[1].get("doc_id") in corrected  # noqa: E731
    i = next((k for k, turn in enumerate(plan) if any(is_corr(c) for c in turn)), None)
    if i is None:
        return plan, None
    first = [c for c in plan[i] if is_corr(c)]
    rest = [c for c in plan[i] if not is_corr(c)]
    if not rest:
        return plan, None
    return plan[:i] + [first, rest] + plan[i + 1:], i


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
                      seed: int = 0, read_recoveries: bool = True, two_phase: bool = False) -> list[dict]:
    """superseded_share: of the train cases with a CORRECTED 1099-INT, how many become
    read_superseded recoveries (v4). The rest stay as clean oracle runs or ordinary recoveries.
    read_recoveries=False keeps only the bad_doc / bad_status recoveries (v3, v5)."""
    rng = random.Random(seed)
    schemas = tool_schemas()
    rows = []
    for cid in case_ids:
        case = env.generate_case(cid)
        kind = None
        if "corrected_1099" in case.notes and rng.random() < superseded_share:
            kind = "read_superseded"
        elif rng.random() < recovery_share:
            decoys = read_recoveries and any(not d.relevant for d in case.documents)
            kinds = ["bad_doc", "bad_status"] + (["read_decoy"] * 2 if decoys else [])
            kind = rng.choice(kinds)
        plan, skip = recovery_plan(cid, rng, kind) if kind else (oracle_plan(cid), None)
        if two_phase:
            plan, split = _split_corrected(cid, plan)
            if skip is not None and split is not None and skip > split:
                skip += 1  # the masked mistake turn moved down by one
        ep = _replay(cid, plan, schemas)
        assert ep["stop"] == "submitted", cid
        # skip_assistant_turns: 0-based assistant-turn indices to exclude from the SFT loss.
        rows.append({"case_id": cid, "kind": kind or "oracle", "messages": ep["messages"],
                     "skip_assistant_turns": [] if skip is None else [skip]})
    return rows


def v5_conversations(n_trap: int = 250, n_decoy: int = 200, n_replay: int = 150, two_phase: bool = False) -> list[dict]:
    """v5, trained on top of v3: clean oracle runs (no wrong reads, even masked), weighted to
    v3's shift failures. n_trap hard cases with a CORRECTED 1099-INT and n_decoy medium/hard
    cases, all from new train ids (700+) that carry the wider train-only decoy pool, plus
    n_replay cases from the original train ids so the rest of the workflow isn't forgotten."""
    trap = [c for c in (f"hard-{i:05d}" for i in range(700, 99_999))
            if "corrected_1099" in env.generate_case(c).notes][:n_trap]
    decoy = [f"{('medium', 'hard')[i % 2]}-{i:05d}" for i in range(20_000, 20_000 + n_decoy)]
    replay = list(SPLITS["train"])
    random.Random(5).shuffle(replay)
    rows = (sft_conversations(trap + decoy, recovery_share=0.1, superseded_share=0, seed=5, read_recoveries=False,
                              two_phase=two_phase)
            + sft_conversations(replay[:n_replay], recovery_share=0.2, superseded_share=0, seed=6,
                                read_recoveries=False, two_phase=two_phase))
    random.Random(7).shuffle(rows)
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
    rows = {"sft": sft_conversations(SPLITS["train"]), "sft-v6": v5_conversations(two_phase=True)}
    for name, rs in rows.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {', '.join(f'{k}={len(v)}' for k, v in SPLITS.items())}, "
          f"{', '.join(f'{k}={len(v)}' for k, v in rows.items())} to {out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2] if len(sys.argv) > 2 else "out")
    else:
        print(__doc__)
