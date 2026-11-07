# TaxOps: teaching a 3B model to file a tax return without wasting tool calls

<!-- DRAFT. {{...}} = fill from the final same-hardware eval (runs/kaggle-eval-final). Keep < 500 words. -->

Agents in production pay for every tool call: a re-read document, an irrelevant file opened, a
malformed call retried. Track 3 problem #2 asks for fewer redundant, looping or failed calls
without lowering task success; problem #1 asks for reliable tool calls. We study both on a task
where success is checkable to the dollar: filing a federal tax return from a folder of documents.

**Environment.** TaxOps generates deterministic synthetic cases from an id: W-2s, 1099-INT/DIV/NEC,
Form 1098, donation receipts and decoys, with six tools (list, read, calculator, standard
deduction, tax computation, submit). Cases contain traps: a CORRECTED 1099-INT that supersedes an
earlier one, the itemize-or-standard choice, and irrelevant documents. Train, eval (300), a
shifted eval with unseen names and unseen decoy types (150), and two dev sets used only for model
selection are disjoint id ranges, checked by tests.

**Method.** We fine-tune Tiny Aya Global (3.35B, Cohere Labs) with LoRA on scripted oracle
demonstrations, then score recorded traces with our verifier and AuditKIT agent_eval. Three
findings shaped the data: Tiny Aya's chat template renders tool-role turns empty, so tool results
go in user turns; the demonstrations must do all arithmetic with the calculator, because a 3B
model's mental sums were the main error; and showing the model *how to recover* after reading the
superseded form (v4) made it open irrelevant documents more often, while showing it *to read the
corrected form first* (v6) halved superseded-form reads. A GRPO stage with a correctness-gated
efficiency reward trained after several fixes (reward variance, vLLM memory and importance
ratios) but ran out of GPU credit at step 21 of 50; we report it as an attempt.

**Metrics.** Success (all 11 fields exact), field accuracy, calls and wasted calls per case,
valid-call rate, AuditKIT precision against the minimal correct run, and tokens per case. Success
alone hides waste and waste alone rewards quitting, so we always report them together.

**Results.** The untrained model never files a correct return (0% on 60 eval cases, 27.8 wasted
of 38.5 calls). The submitted model files {{v6 eval}} of 300 held-out returns exactly with
{{v6 eval wasted}} wasted calls per case and 100% valid calls, and {{v6 shift}} of 150 shifted
cases (v3: {{v3 eval T4}} and {{v3 shift T4}} on the same hardware). Claude Sonnet, given the same
CLI, solves 96.7% of 30 shifted cases but wastes about three times as many calls (0.8 vs 0.23).

**Limits.** The tax rules are a simplified 2025 subset and every case is synthetic. On the shifted
set the v6-over-v3 gain is consistent across metrics but not statistically significant
(20 cases fixed, 12 broken; sign test p ≈ 0.22). The Claude comparison estimates cost from token
counts rather than an API bill.

All code, splits, adapters and every evaluation trace are public, and each number can be
recomputed from the traces.
