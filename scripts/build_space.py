"""Assemble the HF Space (a replay viewer, no GPU) from the published trace folders.

    python scripts/build_space.py --traces <taxops-traces checkout>/traces --out build/space

Copies space/app.py and space/intro.md, the stdlib-only parts of the taxops package, and a
slim copy of each run (case id, stop reason and tool calls; the app rescores them live).
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# (name, column label, test set, trace folder); app.py looks up "v3-<set>" and "v6-<set>" for filters.
RUNS = [
    ("base-eval-shift", "Tiny Aya Global (untrained)", "eval-shift", "base-shift-22"),
    ("v3-eval-shift", "SFT v3", "eval-shift", "sft-v3-shift-150-t4"),
    ("v6-eval-shift", "SFT v6 (submitted)", "eval-shift", "sft-v6-shift-150"),
    ("claude-eval-shift", "Claude Sonnet (closed)", "eval-shift", "claude-sonnet-shift-30"),
    ("base-eval", "Tiny Aya Global (untrained)", "eval", "base-eval-60"),
    ("v3-eval", "SFT v3", "eval", "sft-v3-eval-300-t4"),
    ("v6-eval", "SFT v6 (submitted)", "eval", "sft-v6-eval-300"),
]
README = """---
title: TaxOps replay
emoji: 🧾
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Replay a 3B tax agent's tool calls vs Claude
---
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="build/space")
    args = ap.parse_args()

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "taxops").mkdir(parents=True)
    (out / "traces").mkdir()
    for f in ("app.py", "intro.md"):
        shutil.copy(ROOT / "space" / f, out / f)
    for f in ("__init__.py", "env.py", "metrics.py"):
        shutil.copy(ROOT / "taxops" / f, out / "taxops" / f)
    (out / "README.md").write_text(README)
    (out / "requirements.txt").write_text("")  # gradio comes with the Space SDK

    runs = []
    for name, label, split, folder in RUNS:
        src = Path(args.traces) / folder / "traces.jsonl"
        if not src.exists():
            print(f"skip {name}: no {src}")
            continue
        with open(out / "traces" / f"{name}.jsonl", "w") as f:
            for t in map(json.loads, open(src)):
                f.write(json.dumps({"case_id": t["case_id"], "stop": t.get("stop"), "calls": t["calls"]}) + "\n")
        runs.append({"name": name, "label": label, "split": split, "file": f"{name}.jsonl"})
    (out / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
