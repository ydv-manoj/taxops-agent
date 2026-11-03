"""Run the scripted agents on 50 cases per tier and print the metrics table.

    python3 scripts/demo_scripted.py            # summary table
    python3 scripts/demo_scripted.py hard-00007 # show one case and the oracle's calls
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxops import env  # noqa: E402
from taxops.metrics import score_episode, summarize  # noqa: E402
from taxops.scripted import AGENTS  # noqa: E402


def show_case(case_id: str) -> None:
    case = env.generate_case(case_id)
    print(env.task_prompt(case_id), "\n")
    for d in case.documents:
        print(f"[{d.doc_id}] {'relevant ' if d.relevant else 'DISTRACT '} {d.title}")
    print("\nanswer:", json.dumps(case.answer), "\ntraps:", case.notes or "none", "\n")
    for c in AGENTS["oracle"](case_id):
        print(f"-> {c['name']}({json.dumps(c['arguments'])[:90]})\n   {c['result'][:110]!r}")


def main() -> None:
    if len(sys.argv) > 1:
        return show_case(sys.argv[1])
    cases = [f"{tier}-{i:05d}" for tier in env.TIERS for i in range(50)]
    print(f"{'agent':10} {'success':>8} {'fields':>7} {'calls':>6} {'wasted':>7} {'over min':>9}")
    for name, agent in AGENTS.items():
        s = summarize([score_episode(c, agent(c)) for c in cases])
        over = "-" if s["calls_over_minimum_when_solved"] is None else f"{s['calls_over_minimum_when_solved']:.1f}"
        print(f"{name:10} {s['success_rate']:>8.0%} {s['field_accuracy']:>7.0%} {s['avg_tool_calls']:>6.1f} "
              f"{s['avg_wasted_calls']:>7.1f} {over:>9}")


if __name__ == "__main__":
    main()
