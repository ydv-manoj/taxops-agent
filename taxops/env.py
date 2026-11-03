"""TaxOps: a synthetic tax-preparation tool environment.

Every case is a household's document folder (W-2s, 1099s, a 1098, donation
receipts, plus distractors such as an electric bill or last year's return).
The agent must read what matters, compute a simplified federal return and
submit it. A case is generated deterministically from its id, so the tools
are stateless and any process can rebuild the same case.

Tax rules are a simplified version of 2025 US federal rules (no credits, no
self-employment tax, no adjustments). This is a benchmark, not tax software.
"""

from __future__ import annotations

import ast
import json
import operator
import random
from dataclasses import dataclass, field

STANDARD_DEDUCTION = {"single": 15_750, "married_joint": 31_500}

# (upper bound of bracket, rate). The last bound is infinity.
BRACKETS = {
    "single": [(11_925, 0.10), (48_475, 0.12), (103_350, 0.22), (197_300, 0.24),
               (250_525, 0.32), (626_350, 0.35), (float("inf"), 0.37)],
    "married_joint": [(23_850, 0.10), (96_950, 0.12), (206_700, 0.22), (394_600, 0.24),
                      (501_050, 0.32), (751_600, 0.35), (float("inf"), 0.37)],
}

RETURN_FIELDS = [
    "filing_status", "wages", "interest", "dividends", "business_income", "agi",
    "deduction", "taxable_income", "tax", "withholding", "refund_or_owed",
]

TIERS = ("easy", "medium", "hard")
ALL_TIERS = TIERS + ("shift",)  # "shift" = hard rules with names and decoys never seen in training

EMPLOYERS = ["Acme Logistics Inc", "Brightline Health LLC", "Cedar Software Corp",
             "Delta Foods Co", "Evergreen School District", "Foxglove Retail Inc"]
BANKS = ["First Harbor Bank", "Granite Credit Union", "Summit Savings", "Northwind Bank"]
BROKERS = ["Bluechip Brokerage", "Keystone Investments", "Meridian Funds"]
CHARITIES = ["City Food Bank", "Riverside Animal Shelter", "Literacy Alliance"]
CLIENTS = ["Orbit Design Studio", "Pine Street Cafe", "Quarry Media LLC"]

SHIFT_POOLS = {
    "EMPLOYERS": ["Halcyon Robotics Inc", "Juniper Transit Authority", "Kestrel Pharma Ltd",
                  "Lumen Energy Corp", "Marlow Hospitality Group", "Nimbus Aerospace LLC"],
    "BANKS": ["Ironwood Federal Credit Union", "Lakeview Trust", "Oakmont Savings Bank", "Prairie State Bank"],
    "BROKERS": ["Redwood Capital Markets", "Sterling Index Funds", "Tidewater Securities"],
    "CHARITIES": ["Coastal Cleanup Fund", "Mountain Rescue Volunteers", "Youth Chess League"],
    "CLIENTS": ["Uplift Fitness Studio", "Vantage Print Shop", "Willow Event Planning"],
}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    relevant: bool


@dataclass
class Case:
    case_id: str
    tier: str
    filing_status: str
    documents: list[Document]
    answer: dict
    notes: list[str] = field(default_factory=list)  # traps present in this case

    def doc(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)


def compute_tax_amount(filing_status: str, taxable_income: int) -> int:
    tax, lower = 0.0, 0
    for upper, rate in BRACKETS[filing_status]:
        if taxable_income <= lower:
            break
        tax += (min(taxable_income, upper) - lower) * rate
        lower = upper
    return round(tax)


def _money(x: int) -> str:
    return f"${x:,}"


TRAIN_AUGMENT_BELOW = 100_000  # train ids only; eval ids (100000+) are unchanged


def _augment_train_decoys(docs: list, case_id: str, pools: dict, tax_year: int) -> None:
    """Extra decoys for training cases, of kinds the shift set never uses (shift has a W-4,
    a pay stub and last year's 1099-INT). The prior-year 1099-DIV teaches "check the tax year"
    so the shift set can test whether that carries over to a 1099-INT. Uses its own rng so
    the rest of the case is the same as before."""
    rng = random.Random(case_id + "/train-decoys")
    pool = [
        (f"Form 1099-DIV Dividends - {rng.choice(pools['BROKERS'])} ({tax_year - 1})",
         f"Form 1099-DIV ({tax_year - 1})\nBox 1a Total ordinary dividends: {_money(rng.randint(100, 5_000))}"),
        ("Form W-9 Request for Taxpayer Identification Number",
         "Form W-9\nName: (taxpayer)\nPart I Taxpayer Identification Number: ***-**-4471"),
        (f"Brokerage account summary - {rng.choice(pools['BROKERS'])}",
         f"Account value on Dec 31: {_money(rng.randint(5_000, 90_000))}\n"
         f"Unrealized gain: {_money(rng.randint(100, 9_000))}\n(Not a tax form. See your 1099-DIV.)"),
    ]
    for title, text in rng.sample(pool, rng.randint(0, 2)):
        docs.insert(rng.randint(0, len(docs)), (title, text, False))


def generate_case(case_id: str) -> Case:
    tier = case_id.split("-")[0]
    if tier not in ALL_TIERS:
        raise ValueError(f"case_id must start with one of {ALL_TIERS}: {case_id!r}")
    rng = random.Random(case_id)
    shift = tier == "shift"
    pools = SHIFT_POOLS if shift else {"EMPLOYERS": EMPLOYERS, "BANKS": BANKS, "BROKERS": BROKERS,
                                       "CHARITIES": CHARITIES, "CLIENTS": CLIENTS}
    hard = tier in ("hard", "shift")
    status = rng.choice(["single", "married_joint"])
    docs: list[tuple[str, str, bool]] = []  # (title, text, relevant)
    notes: list[str] = []
    tax_year = 2025

    wages = withholding = interest = dividends = business = mortgage = charity = 0

    n_w2 = 1 if tier == "easy" else rng.randint(1, 2 if status == "single" else 3)
    for employer in rng.sample(pools['EMPLOYERS'], n_w2):
        w = rng.randrange(24_000, 150_000, 25)
        wh = round(w * rng.uniform(0.07, 0.17))
        wages, withholding = wages + w, withholding + wh
        docs.append((f"Form W-2 Wage and Tax Statement - {employer}",
                     f"Form W-2 ({tax_year})\nEmployer: {employer}\n"
                     f"Box 1 Wages, tips, other compensation: {_money(w)}\n"
                     f"Box 2 Federal income tax withheld: {_money(wh)}\n"
                     f"Box 3 Social security wages: {_money(min(w, 176_100))}\n"
                     f"Box 17 State income tax: {_money(round(w * 0.04))}", True))

    if tier != "easy":
        for bank in rng.sample(pools['BANKS'], rng.randint(0, 2)):
            amt = rng.randint(40, 3_200)
            if hard and rng.random() < 0.35:
                # Trap: a corrected form replaces the original from the same payer.
                old = amt + rng.randint(30, 400)
                docs.append((f"Form 1099-INT Interest Income - {bank}",
                             f"Form 1099-INT ({tax_year})\nPayer: {bank}\n"
                             f"Box 1 Interest income: {_money(old)}", False))
                docs.append((f"Form 1099-INT Interest Income - {bank} (CORRECTED)",
                             f"Form 1099-INT ({tax_year}) CORRECTED\nPayer: {bank}\n"
                             f"This form replaces the 1099-INT previously issued.\n"
                             f"Box 1 Interest income: {_money(amt)}", True))
                notes.append("corrected_1099")
            else:
                docs.append((f"Form 1099-INT Interest Income - {bank}",
                             f"Form 1099-INT ({tax_year})\nPayer: {bank}\n"
                             f"Box 1 Interest income: {_money(amt)}", True))
            interest += amt

        if rng.random() < 0.4:
            mortgage = rng.randint(3_000, 24_000)
            docs.append(("Form 1098 Mortgage Interest Statement - " + rng.choice(pools['BANKS']),
                         f"Form 1098 ({tax_year})\nBox 1 Mortgage interest received from payer(s)/borrower(s): "
                         f"{_money(mortgage)}\nBox 2 Outstanding mortgage principal: "
                         f"{_money(rng.randint(150, 600) * 1_000)}", True))
        for org in rng.sample(pools['CHARITIES'], rng.randint(0, 2)):
            gift = rng.randint(100, 9_000)
            charity += gift
            docs.append((f"Donation receipt - {org}",
                         f"{org}\nThank you for your cash gift of {_money(gift)} received in {tax_year}.\n"
                         f"No goods or services were provided in exchange.", True))

    if hard:
        if rng.random() < 0.6:
            broker = rng.choice(pools['BROKERS'])
            dividends = rng.randint(100, 6_000)
            docs.append((f"Form 1099-DIV Dividends - {broker}",
                         f"Form 1099-DIV ({tax_year})\nPayer: {broker}\n"
                         f"Box 1a Total ordinary dividends: {_money(dividends)}\n"
                         f"Box 1b Qualified dividends: {_money(round(dividends * 0.6))}", True))
        if rng.random() < 0.5:
            client = rng.choice(pools['CLIENTS'])
            gross = rng.randint(5_000, 60_000)
            expenses = round(gross * rng.uniform(0.1, 0.45))
            business = gross - expenses
            docs.append((f"Form 1099-NEC Nonemployee Compensation - {client}",
                         f"Form 1099-NEC ({tax_year})\nPayer: {client}\n"
                         f"Box 1 Nonemployee compensation: {_money(gross)}", True))
            docs.append(("Freelance expense ledger",
                         f"Business expenses for {tax_year} (freelance work)\n"
                         f"Software and subscriptions: {_money(round(expenses * 0.3))}\n"
                         f"Equipment: {_money(round(expenses * 0.5))}\n"
                         f"Home office supplies: {_money(expenses - round(expenses * 0.3) - round(expenses * 0.5))}\n"
                         f"Total business expenses: {_money(expenses)}", True))

    distractors = [
        ("Electric bill - October", f"City Power & Light\nAmount due: {_money(rng.randint(60, 240))}\nDue date: Nov 15"),
        (f"Form 1040 for {tax_year - 1} (prior year copy)",
         f"U.S. Individual Income Tax Return {tax_year - 1}\nLine 1a Wages: {_money(rng.randrange(20_000, 140_000, 25))}\n"
         f"Line 15 Taxable income: {_money(rng.randrange(10_000, 120_000, 25))}\nLine 24 Total tax: {_money(rng.randint(1_000, 20_000))}"),
        ("Bank statement - December",
         f"{rng.choice(pools['BANKS'])}\nOpening balance: {_money(rng.randint(1_000, 30_000))}\n"
         f"Deposits: {_money(rng.randint(2_000, 15_000))}\nInterest paid this month: {_money(rng.randint(1, 60))}"),
        ("Health insurance card", "Member ID: XJ4-22918\nGroup: 4471\nPlan: Silver PPO"),
    ]
    if shift:  # decoys of kinds never seen in training, including last year's 1099
        distractors += [
            ("Form W-4 Employee's Withholding Certificate",
             f"Form W-4 ({tax_year})\nStep 4(c) Extra withholding per pay period: {_money(rng.randint(20, 300))}\n"
             f"Step 3 Claim dependents: {_money(rng.choice([0, 2_000, 4_000]))}"),
            (f"Form 1099-INT Interest Income - {rng.choice(pools['BANKS'])} ({tax_year - 1})",
             f"Form 1099-INT ({tax_year - 1})\nBox 1 Interest income: {_money(rng.randint(40, 2_500))}"),
            ("Pay stub - December",
             f"Gross pay this period: {_money(rng.randint(2_000, 9_000))}\nYear-to-date federal tax: "
             f"{_money(rng.randint(3_000, 20_000))}\n(Use your W-2 for tax filing.)"),
        ]
    n_distract = {"easy": 0, "medium": 1, "hard": rng.randint(2, 3), "shift": rng.randint(3, 4)}[tier]
    for title, text in rng.sample(distractors, n_distract):
        docs.append((title, text, False))

    rng.shuffle(docs)
    if not shift and tier != "easy" and int(case_id.rsplit("-", 1)[1]) < TRAIN_AUGMENT_BELOW:
        _augment_train_decoys(docs, case_id, pools, tax_year)
    documents = [Document(f"D{i + 1}", t, x, r) for i, (t, x, r) in enumerate(docs)]

    agi = wages + interest + dividends + business
    deduction = max(STANDARD_DEDUCTION[status], mortgage + charity)
    if mortgage + charity > STANDARD_DEDUCTION[status]:
        notes.append("itemizes")
    taxable = max(0, agi - deduction)
    tax = compute_tax_amount(status, taxable)
    answer = {
        "filing_status": status, "wages": wages, "interest": interest, "dividends": dividends,
        "business_income": business, "agi": agi, "deduction": deduction, "taxable_income": taxable,
        "tax": tax, "withholding": withholding, "refund_or_owed": withholding - tax,
    }
    return Case(case_id, tier, status, documents, answer, notes)


def task_prompt(case_id: str) -> str:
    case = generate_case(case_id)
    status = "single" if case.filing_status == "single" else "married filing jointly"
    return (
        f"Prepare the 2025 federal return for case {case_id}. The taxpayer files as {status}. "
        "Find the relevant documents, compute every return field, then call submit_return once. "
        "Rules: AGI = wages + interest + ordinary dividends + business income (1099-NEC minus business "
        "expenses). Deduction = the larger of the standard deduction and itemized deductions "
        "(mortgage interest + cash charity). Taxable income = AGI - deduction, not below 0. "
        "If a form is marked CORRECTED, it replaces the original. refund_or_owed = withholding - tax "
        "(negative means the taxpayer owes). Use whole dollars."
    )


# ---------------------------------------------------------------------------
# Tools. Plain functions with type hints and Google-style docstrings, which is
# what AgentTune turns into tool schemas. Errors are returned as text starting
# with "ERROR:" so the agent can recover, and so metrics can count them.
# ---------------------------------------------------------------------------

def _load(case_id: str) -> Case | str:
    try:
        return generate_case(str(case_id).strip())
    except (ValueError, IndexError):
        return f"ERROR: unknown case_id {case_id!r}."


def list_documents(case_id: str) -> str:
    """List the documents in a taxpayer's folder.

    Args:
        case_id: the case to open, for example "medium-00012"
    """
    case = _load(case_id)
    if isinstance(case, str):
        return case
    return json.dumps([{"doc_id": d.doc_id, "title": d.title} for d in case.documents])


def read_document(case_id: str, doc_id: str) -> str:
    """Read the full text of one document.

    Args:
        case_id: the case the document belongs to
        doc_id: the document id from list_documents, for example "D3"
    """
    case = _load(case_id)
    if isinstance(case, str):
        return case
    doc = case.doc(str(doc_id).strip())
    if doc is None:
        return f"ERROR: no document {doc_id!r} in case {case_id}. Call list_documents to see valid ids."
    return doc.text


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with + - * / and parentheses.

    Args:
        expression: for example "52310 + 1480 - 31500"
    """
    try:
        value = _eval(ast.parse(str(expression).replace(",", "").replace("$", ""), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"ERROR: could not evaluate {expression!r} ({e})."
    return str(round(value, 2))


def _status(filing_status: str) -> str | None:
    s = str(filing_status).strip().lower().replace(" ", "_")
    return {"single": "single", "married_joint": "married_joint", "married_filing_jointly": "married_joint",
            "mfj": "married_joint"}.get(s)


def lookup_standard_deduction(filing_status: str) -> str:
    """Get the 2025 standard deduction for a filing status.

    Args:
        filing_status: "single" or "married_joint"
    """
    s = _status(filing_status)
    if s is None:
        return f"ERROR: unknown filing_status {filing_status!r}. Use 'single' or 'married_joint'."
    return str(STANDARD_DEDUCTION[s])


def compute_tax(filing_status: str, taxable_income: int) -> str:
    """Compute 2025 federal income tax from the tax brackets.

    Args:
        filing_status: "single" or "married_joint"
        taxable_income: taxable income in whole dollars
    """
    s = _status(filing_status)
    if s is None:
        return f"ERROR: unknown filing_status {filing_status!r}. Use 'single' or 'married_joint'."
    try:
        income = int(round(float(str(taxable_income).replace(",", "").replace("$", ""))))
    except ValueError:
        return f"ERROR: taxable_income must be a number, got {taxable_income!r}."
    return str(compute_tax_amount(s, max(0, income)))


def submit_return(case_id: str, filing_status: str, wages: int, interest: int, dividends: int,
                  business_income: int, agi: int, deduction: int, taxable_income: int, tax: int,
                  withholding: int, refund_or_owed: int) -> str:
    """Submit the finished return. Call this once, at the end.

    Args:
        case_id: the case being filed
        filing_status: "single" or "married_joint"
        wages: total W-2 box 1 wages
        interest: total taxable interest
        dividends: total ordinary dividends
        business_income: 1099-NEC income minus business expenses
        agi: adjusted gross income
        deduction: standard or itemized deduction, whichever is larger
        taxable_income: agi minus deduction, not below 0
        tax: federal income tax on taxable_income
        withholding: total federal income tax withheld
        refund_or_owed: withholding minus tax
    """
    case = _load(case_id)
    if isinstance(case, str):
        return case
    # Deliberately no score in the reply: the agent can't probe its way to the answer.
    return f"Return for case {case.case_id} received."


TOOLS = [list_documents, read_document, calculator, lookup_standard_deduction, compute_tax, submit_return]
