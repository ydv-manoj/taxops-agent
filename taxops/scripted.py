"""Scripted agents that drive the real tools. They prove the environment and the
metrics behave before any model is involved:

- oracle:   reads only what matters, minimum calls, always right
- wasteful: right answer, but re-lists, re-reads, opens distractors, typos an id
- careless: efficient, but falls for the traps (counts the original 1099-INT
            next to its CORRECTED copy, always takes the standard deduction)
"""

from __future__ import annotations

from taxops import env


def _call(log: list[dict], fn, **args) -> str:
    result = fn(**args)
    log.append({"name": fn.__name__, "arguments": args, "result": result})
    return result


def _finish(log: list[dict], case: env.Case, answer: dict) -> list[dict]:
    _call(log, env.lookup_standard_deduction, filing_status=case.filing_status)
    _call(log, env.compute_tax, filing_status=case.filing_status, taxable_income=answer["taxable_income"])
    _call(log, env.submit_return, case_id=case.case_id, **answer)
    return log


def oracle(case_id: str) -> list[dict]:
    case, log = env.generate_case(case_id), []
    _call(log, env.list_documents, case_id=case_id)
    for d in case.documents:
        if d.relevant:
            _call(log, env.read_document, case_id=case_id, doc_id=d.doc_id)
    return _finish(log, case, case.answer)


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
