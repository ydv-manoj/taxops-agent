"""Command-line access to the TaxOps tools, for the Claude comparison run.

    python -m taxops.cli prompt hard-100004
    python -m taxops.cli call hard-100004 '{"tool_name": "list_documents", "parameters": {"case_id": "hard-100004"}}'
    python -m taxops.cli score hard-100004

Every call is appended to runs/claude/calls/<case_id>.jsonl, exactly as the agent
loop records them, so Claude's episodes are scored the same way as Tiny Aya's.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from taxops import env
from taxops.agent_loop import execute, parse_calls

LOG_DIR = Path(os.environ.get("TAXOPS_LOG_DIR", Path(__file__).resolve().parents[1] / "runs" / "claude" / "calls"))


def _log(case_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{case_id}.jsonl"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"prompt", "call", "score"}:
        print(__doc__)
        return 2
    cmd, case_id = argv[0], argv[1]
    if cmd == "prompt":
        from taxops.agent_loop import system_prompt, tool_schemas

        # Same system prompt and task text Tiny Aya gets.
        print("SYSTEM:\n" + system_prompt(tool_schemas()) + "\n\nTASK:\n" + env.task_prompt(case_id))
        return 0
    if cmd == "score":
        from taxops.metrics import score_episode

        path = _log(case_id)
        calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        print(json.dumps(score_episode(case_id, calls).to_dict()))
        return 0
    calls = parse_calls(argv[2] if len(argv) > 2 else "")
    if calls is None:
        print('ERROR: pass a JSON list like [{"tool_name": "...", "parameters": {...}}].')
        calls = [{"name": "unparseable", "arguments": None}]
        calls[0]["result"] = "ERROR: unparseable call"
        with open(_log(case_id), "a") as f:
            f.write(json.dumps(calls[0]) + "\n")
        return 1
    with open(_log(case_id), "a") as f:
        for call in calls:
            call["result"] = execute(call)
            f.write(json.dumps(call) + "\n")
            print(f"Tool result for {call['name']}: {call['result']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
