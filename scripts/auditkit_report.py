"""Score saved episodes with AuditKIT: our verified outcome as the headline, plus
AuditKIT's own tool-use diagnostics as an independent check.

    python scripts/auditkit_report.py runs/base/traces.jsonl
    python scripts/auditkit_report.py --demo        # scripted oracle vs wasteful agents

Writes <traces dir>/auditkit.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auditkit.agent_eval import (AgentCase, AgentEvalRunner, AgentEvalSpec, CustomPredicate,  # noqa: E402
                                 episode_from_openai_messages)

from taxops import env  # noqa: E402
from taxops.metrics import score_episode  # noqa: E402
from taxops.scripted import AGENTS, oracle  # noqa: E402

# tool_selection is left out: it needs an LLM judge.
SCORERS = ["tool_call_validity", "tool_call_f1_name", "redundant_tool_calls", "agent_loop_detection"]
DIAGNOSTICS = {"tool_call_validity": "valid_call_rate", "tool_call_precision_name": "precision_vs_oracle",
               "redundant_tool_calls": "redundant_share", "agent_loop_detection": "no_loop_rate"}


def to_openai(case_id: str, calls: list[dict]) -> list[dict]:
    msgs = [{"role": "user", "content": env.task_prompt(case_id)}]
    for i, c in enumerate(calls):
        cid = f"call_{i}"
        args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": c.get("name", ""), "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": str(c.get("result", ""))})
    return msgs


def verified_return(episode, case) -> bool:
    """Headline outcome: the submitted return matches the computed answer on all 11 fields."""
    return score_episode(case.id, episode.metadata["taxops_calls"]).success


def build(rows: list[dict]):
    episodes, cases = [], []
    names = [t.__name__ for t in env.TOOLS]
    for r in rows:
        cid, calls = r["case_id"], r["calls"]
        ep = episode_from_openai_messages(to_openai(cid, calls), case_id=cid)
        ep.metadata["taxops_calls"] = calls
        episodes.append(ep)
        reference = [[{"name": c["name"], "arguments": c["arguments"]}] for c in oracle(cid)]
        cases.append(AgentCase(id=cid, task=env.task_prompt(cid), category="tool_use", allowed_tools=names,
                               reference_turns=reference,
                               outcome=CustomPredicate(verified_return, version="taxops-verifier-v1")))
    return episodes, cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        ids = [f"{t}-{i:05d}" for t in env.TIERS for i in range(100_000, 100_010)]
        runs = {name: [{"case_id": c, "calls": agent(c)} for c in ids] for name, agent in AGENTS.items()}
    else:
        rows = [json.loads(line) for line in open(args.traces)]
        runs = {Path(args.traces).parent.name: rows}

    report = {}
    for name, rows in runs.items():
        episodes, cases = build(rows)
        result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                                                     scorers=SCORERS, trials=1, name=f"taxops-{name}"))
        verdicts = [row.outcome["verdict"] for row in result.rows]
        means = {}
        for key, label in DIAGNOSTICS.items():
            vals = [row.diagnostics[key]["value"] for row in result.rows
                    if key in row.diagnostics and row.diagnostics[key]["value"] is not None]
            means[label] = round(sum(vals) / len(vals), 3) if vals else None
        report[name] = {"verified_success_rate": verdicts.count("success") / len(verdicts), **means,
                        "summary": result.summary()}
        print(f"{name:10} verified success {report[name]['verified_success_rate']:.0%}  " +
              "  ".join(f"{k} {v}" for k, v in means.items()))
    if args.traces:
        (Path(args.traces).parent / "auditkit.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
