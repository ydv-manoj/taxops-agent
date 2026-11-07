"""Merge eval shards (eval_model.py --shard i/n) into one run with a recomputed summary.

    python scripts/merge_shards.py --out runs/sft-v6-shift runs/kaggle/sft-v6-shift-0 runs/kaggle/sft-v6-shift-1

Scores are recomputed from the traces with taxops.metrics, so the merged summary doesn't trust
the per-shard numbers.
"""

import argparse
import json
from pathlib import Path

from taxops.agent_loop import parse_calls
from taxops.metrics import score_episode, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    traces, parts = [], []
    for d in map(Path, args.shards):
        traces += [json.loads(line) for line in open(d / "traces.jsonl")]
        if (d / "summary.json").exists():
            parts.append(json.loads((d / "summary.json").read_text()))
    ids = [t["case_id"] for t in traces]
    assert len(ids) == len(set(ids)), "shards overlap"

    scores = [score_episode(t["case_id"], t["calls"]) for t in traces]
    replies = [m["content"] for t in traces for m in t["messages"] if m["role"] == "assistant"]
    n = max(1, len(traces))
    first = parts[0] if parts else {}
    summary = {"model": first.get("model"), "adapter": first.get("adapter"), "split": first.get("split"),
               "shards": [p.get("shard") for p in parts], "overall": summarize(scores),
               "by_tier": {t: summarize([s for s in scores if s.case_id.startswith(t + "-")])
                           for t in sorted({s.case_id.rsplit("-", 1)[0] for s in scores})},
               "valid_call_rate": sum(parse_calls(r) is not None for r in replies) / max(1, len(replies)),
               "avg_seconds_per_case": sum(t["seconds"] for t in traces) / n,
               "avg_tokens_per_case": {k: sum(t["tokens"][k] for t in traces) / n
                                       for k in ("prompt_tokens", "completion_tokens")},
               "gpu": first.get("gpu")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "traces.jsonl", "w") as f:
        for t in sorted(traces, key=lambda t: t["case_id"]):
            f.write(json.dumps(t) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(traces)} cases -> {out}", json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
