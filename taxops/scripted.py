"""Scripted agents that drive the real tools. They prove the environment and the
metrics behave before any model is involved:

- oracle:   reads only what matters, minimum calls, always right
- wasteful: right answer, but re-lists, re-reads, opens distractors, typos an id
- careless: efficient, but falls for the traps (counts the original 1099-INT
            next to its CORRECTED copy, always takes the standard deduction)
"""

from __future__ import annotations

import re

from taxops import env


def _call(log: list[dict], fn, turn: int | None = None, **args) -> str:
    result = fn(**args)
    entry = {"name": fn.__name__, "arguments": args, "result": result}
    if turn is not None:
        entry["turn"] = turn  # calls with the same turn are sent together in one reply
    log.append(entry)
    return result


def _amounts(text: str, label: str) -> list[int]:
    return [int(m.replace(",", "")) for m in re.findall(re.escape(label) + r"[^$\n]*\$([\d,]+)", text)]


def _finish(log: list[dict], case: env.Case, answer: dict) -> list[dict]:
    _call(log, env.lookup_standard_deduction, filing_status=case.filing_status)
    _call(log, env.compute_tax, filing_status=case.filing_status, taxable_income=answer["taxable_income"])
    _call(log, env.submit_return, case_id=case.case_id, **answer)
    return log


def oracle(case_id: str) -> list[dict]:
    """Minimal, correct run that does every sum with the calculator (a 3B model's mental
    arithmetic is the main error source), batching independent calls into one turn:
    list -> read all relevant docs -> sums + standard deduction -> taxable -> tax -> refund -> submit."""
    case, log, a = env.generate_case(case_id), [], env.generate_case(case_id).answer
    _call(log, env.list_documents, turn=0, case_id=case_id)
    relevant = [d for d in case.documents if d.relevant]
    for d in relevant:
        _call(log, env.read_document, turn=1, case_id=case_id, doc_id=d.doc_id)

    wages, withheld, income, itemized, business = [], [], [], [], None
    for d in relevant:
        t = d.text
        if t.startswith("Form W-2"):
            wages += _amounts(t, "Box 1 Wages")
            withheld += _amounts(t, "Box 2 Federal income tax withheld")
        elif t.startswith("Form 1099-INT"):
            income += _amounts(t, "Box 1 Interest income")
        elif t.startswith("Form 1099-DIV"):
            income += _amounts(t, "Box 1a Total ordinary dividends")
        elif t.startswith("Form 1098"):
            itemized += _amounts(t, "Box 1 Mortgage interest")
        elif "cash gift of" in t:
            itemized += _amounts(t, "cash gift of")
        elif t.startswith("Form 1099-NEC"):
            business = f"{_amounts(t, 'Box 1 Nonemployee compensation')[0]}"
        elif t.startswith("Business expenses"):
            expenses = _amounts(t, "Total business expenses")[0]
    if business is not None:
        business = f"{business} - {expenses}"

    def calc(turn: int, terms: list[str]) -> None:
        if len(terms) > 1 or (terms and " - " in terms[0]):
            _call(log, env.calculator, turn=turn, expression=" + ".join(terms))

    agi_terms = [str(x) for x in wages + income] + ([f"({business})"] if business else [])
    if business:
        calc(2, [business])
    calc(2, agi_terms)
    calc(2, [str(x) for x in withheld])
    calc(2, [str(x) for x in itemized])
    _call(log, env.lookup_standard_deduction, turn=2, filing_status=case.filing_status)
    _call(log, env.calculator, turn=3, expression=f"{a['agi']} - {a['deduction']}")
    _call(log, env.compute_tax, turn=4, filing_status=case.filing_status, taxable_income=a["taxable_income"])
    _call(log, env.calculator, turn=5, expression=f"{a['withholding']} - {a['tax']}")
    _call(log, env.submit_return, turn=6, case_id=case_id, **a)
    return log


def wasteful(case_id: str) -> list[dict]:
    case, log = env.generate_case(case_id), []
    _call(log, env.list_documents, case_id=case_id)
    _call(log, env.read_document, case_id=case_id, doc_id="D99")  # failed call
    _call(log, env.list_documents, case_id=case_id)               # redundant
    for d in case.documents:
        _call(log, env.read_document, case_id=case_id, doc_id=d.doc_id)  # distractors too
    _call(log, env.read_document, case_id=case_id, doc_id=case.documents[0].doc_id)  # re-read
    return _finish(log, case, case.answer)


def careless(case_id: str) -> list[dict]:
    case, log = env.generate_case(case_id), []
    _call(log, env.list_documents, case_id=case_id)
    for d in case.documents:
        if d.relevant:
            _call(log, env.read_document, case_id=case_id, doc_id=d.doc_id)
    a = dict(case.answer)
    if "corrected_1099" in case.notes:
        # Adds the superseded original too (its amount isn't in the answer, so approximate the mistake).
        a["interest"] += 100
    a["deduction"] = env.STANDARD_DEDUCTION[case.filing_status]
    a["agi"] = a["wages"] + a["interest"] + a["dividends"] + a["business_income"]
    a["taxable_income"] = max(0, a["agi"] - a["deduction"])
    a["tax"] = env.compute_tax_amount(case.filing_status, a["taxable_income"])
    a["refund_or_owed"] = a["withholding"] - a["tax"]
    return _finish(log, case, a)


AGENTS = {"oracle": oracle, "wasteful": wasteful, "careless": careless}
