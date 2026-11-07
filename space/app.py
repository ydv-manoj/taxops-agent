"""TaxOps replay: pick a case and compare recorded agent runs side by side.

Nothing runs a model here. Every run is a recorded trace from the TaxOps evals, rescored live
with the same verifier (taxops.metrics) that produced the published numbers.
"""

import json
from pathlib import Path

import gradio as gr

from taxops.env import RETURN_FIELDS, TOOLS, generate_case
from taxops.metrics import _key, _same, score_episode

HERE = Path(__file__).parent
TOOL_NAMES = {t.__name__ for t in TOOLS}
RUNS = json.loads((HERE / "runs.json").read_text())  # [{"name", "label", "split", "file"}]


def load(file: str) -> dict:
    return {t["case_id"]: t for t in map(json.loads, open(HERE / "traces" / file))}


TRACES = {r["name"]: load(r["file"]) for r in RUNS}
SPLITS = sorted({r["split"] for r in RUNS})


def label_calls(case_id: str, calls: list[dict]) -> list[str]:
    """Why each call counts as waste, mirroring score_episode; "" for a useful call."""
    case = generate_case(case_id)
    seen, submitted, out = set(), False, []
    for call in calls:
        name, args = call.get("name"), call.get("arguments") or {}
        if name not in TOOL_NAMES or not isinstance(args, dict):
            out.append("invalid call")
            continue
        if str(call.get("result", "")).startswith("ERROR"):
            out.append("failed call")
            continue
        if submitted:
            out.append("after submitting")
            continue
        key, why = _key(call), ""
        if key in seen or (name == "list_documents" and any(k.startswith("list_documents") for k in seen)):
            why = "repeat"
        elif name == "read_document" and (doc := case.doc(str(args.get("doc_id", "")).strip())) and not doc.relevant:
            why = "superseded form" if "1099-INT" in doc.title and "corrected_1099" in case.notes \
                and "(20" not in doc.title else f"irrelevant: {doc.title}"
        seen.add(key)
        submitted = submitted or name == "submit_return"
        out.append(why)
    return out


def describe(call: dict) -> str:
    args = dict(call.get("arguments") or {})
    args.pop("case_id", None)
    if call.get("name") == "read_document":
        return f"read_document {args.get('doc_id', '')}"
    if call.get("name") == "submit_return":
        return "submit_return"
    return f"{call.get('name')} {json.dumps(args)[:80]}" if args else str(call.get("name"))


def render(run: dict, case_id: str) -> str:
    t = TRACES[run["name"]].get(case_id)
    if t is None:
        return f"### {run['label']}\n\n_not run on this case_"
    calls = t["calls"]
    s = score_episode(case_id, calls)
    head = "✅ correct return" if s.success else ("❌ wrong return" if s.submitted else "⛔ never submitted")
    lines = [f"### {run['label']}", f"**{head}** · {round(s.field_accuracy * 11)}/11 fields · "
             f"{s.tool_calls} calls · **{s.wasted_calls} wasted**", ""]
    for i, (call, why) in enumerate(zip(calls, label_calls(case_id, calls)), 1):
        tag = f" — ⚠️ *{why}*" if why else ""
        lines.append(f"{i}. `{describe(call)}`{tag}")
    sub = next((c.get("arguments") for c in calls if c.get("name") == "submit_return"), None)
    if isinstance(sub, dict):
        case = generate_case(case_id)
        wrong = [f"{f}: {sub.get(f)} (want {case.answer[f]})" for f in RETURN_FIELDS
                 if not (f in sub and _same(sub[f], case.answer[f]))]
        if wrong:
            lines += ["", "**Wrong fields:** " + "; ".join(wrong)]
    return "\n".join(lines)


def case_ids(split: str, only: str) -> list[str]:
    runs = [r for r in RUNS if r["split"] == split]
    ids = sorted(set().union(*(TRACES[r["name"]] for r in runs)))
    ok = lambda name, c: name in TRACES and c in TRACES[name] and score_episode(c, TRACES[name][c]["calls"]).success
    if only == "v6 fixed, v3 failed":
        ids = [c for c in ids if ok(f"v6-{split}", c) and not ok(f"v3-{split}", c)]
    elif only == "v6 failed":
        ids = [c for c in ids if f"v6-{split}" in TRACES and c in TRACES[f"v6-{split}"] and not ok(f"v6-{split}", c)]
    elif only == "has base + Claude runs":
        ids = [c for c in ids if all(c in TRACES[r["name"]] for r in runs)]
    return ids


def folder(case_id: str) -> str:
    case = generate_case(case_id)
    rows = [f"| {d.doc_id} | {d.title} | {'✓' if d.relevant else '—'} |" for d in case.documents]
    traps = ", ".join(case.notes) or "none"
    return (f"**{case_id}** · filing status: {case.answer['filing_status']} · traps: {traps}\n\n"
            "| doc | title | needed |\n|---|---|---|\n" + "\n".join(rows))


def show(split: str, case_id: str):
    runs = [r for r in RUNS if r["split"] == split]
    outs = [render(r, case_id) if case_id else "" for r in runs]
    outs += [""] * (MAX_COLS - len(outs))
    return [folder(case_id) if case_id else ""] + outs


MAX_COLS = max(sum(r["split"] == s for r in RUNS) for s in SPLITS)
FILTERS = ["all", "v6 fixed, v3 failed", "v6 failed", "has base + Claude runs"]

with gr.Blocks(title="TaxOps replay") as demo:
    gr.Markdown((HERE / "intro.md").read_text())
    with gr.Row():
        split = gr.Radio(SPLITS, value=SPLITS[-1], label="Test set")
        only = gr.Radio(FILTERS, value="has base + Claude runs", label="Show cases")
        case = gr.Dropdown([], label="Case")
    docs = gr.Markdown()
    with gr.Row():
        cols = [gr.Markdown() for _ in range(MAX_COLS)]

    def pick(split, only):
        ids = case_ids(split, only)
        return gr.Dropdown(choices=ids, value=ids[0] if ids else None)

    for ev in (split.change, only.change):
        ev(pick, [split, only], case)
    case.change(show, [split, case], [docs] + cols)
    demo.load(pick, [split, only], case)

if __name__ == "__main__":
    demo.launch()
