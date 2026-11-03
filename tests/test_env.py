import json

from taxops import env
from taxops.metrics import score_episode
from taxops.scripted import careless, oracle, wasteful

CASES = [f"{t}-{i:05d}" for t in env.TIERS for i in range(40)]


def test_cases_are_deterministic():
    a, b = env.generate_case("hard-00003"), env.generate_case("hard-00003")
    assert a.answer == b.answer and [d.text for d in a.documents] == [d.text for d in b.documents]


def test_tax_brackets():
    assert env.compute_tax_amount("single", 0) == 0
    assert env.compute_tax_amount("single", 11_925) == 1_192  # 10% of the first bracket, rounded
    assert env.compute_tax_amount("single", 50_000) == round(1_192.5 + (48_475 - 11_925) * 0.12 + 1_525 * 0.22)


def test_answer_is_consistent():
    for cid in CASES:
        a = env.generate_case(cid).answer
        assert a["agi"] == a["wages"] + a["interest"] + a["dividends"] + a["business_income"]
        assert a["taxable_income"] == max(0, a["agi"] - a["deduction"])
        assert a["refund_or_owed"] == a["withholding"] - a["tax"]


def test_hard_tier_has_traps():
    notes = [n for cid in CASES if cid.startswith("hard") for n in env.generate_case(cid).notes]
    assert "corrected_1099" in notes and "itemizes" in notes


def test_tool_errors_are_recoverable_text():
    assert env.read_document("easy-00001", "D99").startswith("ERROR")
    assert env.list_documents("nope-1").startswith("ERROR")
    assert env.calculator("__import__('os')").startswith("ERROR")
    assert env.compute_tax("head_of_household", 1000).startswith("ERROR")
    assert env.calculator("$1,000 + 250") == "1250"
    assert json.loads(env.list_documents("easy-00001"))[0]["doc_id"] == "D1"


def test_oracle_is_perfect_and_minimal():
    for cid in CASES:
        s = score_episode(cid, oracle(cid))
        assert s.success and s.wasted_calls == 0 and s.tool_calls == s.min_calls, cid


def test_wasteful_is_right_but_wastes():
    for cid in CASES:
        s = score_episode(cid, wasteful(cid))
        assert s.success and s.failed_calls == 1 and s.redundant_calls >= 2, cid


def test_careless_fails_on_traps():
    hard = [score_episode(c, careless(c)) for c in CASES if c.startswith("hard")]
    assert any(not s.success for s in hard) and all(s.wasted_calls == 0 for s in hard)


def test_no_submit_means_no_credit():
    calls = oracle("medium-00002")[:-1]
    s = score_episode("medium-00002", calls)
    assert not s.submitted and s.field_accuracy == 0


def test_strings_and_status_aliases_accepted():
    cid = "easy-00004"
    calls = oracle(cid)
    sub = calls[-1]["arguments"]
    sub.update({k: f"${v:,}" for k, v in sub.items() if isinstance(v, int)})
    if sub["filing_status"] == "married_joint":
        sub["filing_status"] = "Married filing jointly"
    assert score_episode(cid, calls).success
