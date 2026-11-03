"""Run one episode with any text generator, in the same message format AgentTune
uses for Tiny Aya (tool schemas in the system prompt, a JSON list of calls in
the reply, tool results folded back as user turns). Training passes the same
TOOLS_PROMPT and RESULT_FORMAT to AgentTune, so before/after evals match.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from taxops import env

TOOLS_PROMPT = (
    "You are a tax preparer. You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call. '
    "Keep calling tools until you have called submit_return; that ends the task."
)
RESULT_FORMAT = "Tool result for {name}: {content}"

MAX_STEPS = 20


def tool_schemas() -> list[dict]:
    from transformers.utils import get_json_schema  # same schema builder AgentTune uses

    return [get_json_schema(t) for t in env.TOOLS]


def system_prompt(schemas: list[dict]) -> str:
    return TOOLS_PROMPT.format(tools=json.dumps(schemas, indent=2))


_ACTION = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.S)


def parse_calls(text: str) -> list[dict] | None:
    """Pull tool calls out of a reply. Returns None when the reply has no parseable calls."""
    m = _ACTION.search(text)
    candidates = [m.group(1)] if m else []
    start = text.find("[")
    if start != -1:
        candidates.append(text[start:text.rfind("]") + 1])
    start = text.find("{")
    if start != -1:
        candidates.append("[" + text[start:text.rfind("}") + 1] + "]")
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        calls = []
        for d in data if isinstance(data, list) else []:
            if isinstance(d, dict):
                name = d.get("tool_name") or d.get("name")
                args = d.get("parameters", d.get("arguments", {}))
                if name:
                    calls.append({"name": str(name), "arguments": args})
        if calls:
            return calls
    return None


_BY_NAME = {t.__name__: t for t in env.TOOLS}


def execute(call: dict) -> str:
    fn = _BY_NAME.get(call["name"])
    if fn is None:
        return f"ERROR: unknown tool {call['name']!r}. Available: {', '.join(_BY_NAME)}."
    if not isinstance(call["arguments"], dict):
        return "ERROR: parameters must be a JSON object."
    try:
        return fn(**call["arguments"])
    except TypeError as e:
        return f"ERROR: bad parameters for {call['name']}: {e}."


def run_episode(case_id: str, generate: Callable[[list[dict]], str], schemas: list[dict]) -> dict:
    """generate(messages) -> assistant text. Returns the transcript and every tool call."""
    messages = [{"role": "system", "content": system_prompt(schemas)},
                {"role": "user", "content": env.task_prompt(case_id)}]
    calls: list[dict] = []
    stop = "max_steps"
    for _ in range(MAX_STEPS):
        reply = generate(messages)
        messages.append({"role": "assistant", "content": reply})
        parsed = parse_calls(reply)
        if parsed is None:
            stop = "no_tool_call"  # the model answered in prose; the episode ends
            break
        results = []
        for call in parsed:
            call["result"] = execute(call)
            calls.append(call)
            results.append(RESULT_FORMAT.format(name=call["name"], content=call["result"]))
        messages.append({"role": "user", "content": "\n".join(results)})
        if any(c["name"] == "submit_return" for c in parsed):
            stop = "submitted"
            break
    return {"case_id": case_id, "stop": stop, "calls": calls, "messages": messages}
