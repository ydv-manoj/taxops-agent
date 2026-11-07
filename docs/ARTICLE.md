---
title: "TaxOps: teaching a 3B model to file a tax return without wasting tool calls"
authors: Manoj Kumar
---

<!-- DRAFT for the HF Article. -->

## TL;DR

We fine-tuned **Tiny Aya Global (3.35B)** to work through a folder of synthetic tax documents
with six tools and file a federal return. The untrained model never files a correct return and
spends about 28 of its ~38 calls on waste. After supervised fine-tuning on scripted
demonstrations (AgentTune pipeline, scored with AuditKIT), it files **88.7%** of 300 held-out
returns exactly right with **0.06 wasted calls per case**, and **68.7%** on a harder shifted set it
never saw. Claude Sonnet, run the same way, is more accurate on the shifted set (96.7%) but wastes
about three times as many calls. We also report what didn't work: a GRPO stage that ran out of
GPU credit at step 21, and a handful of tooling bugs that cost us days and may save you some.

## Why this problem

In production agent workflows (the kind Manoj builds for tax work, where a run takes ~20 minutes), the
bill is mostly tool calls and the tokens they drag into the context: re-reading a document,
opening a file that doesn't matter, retrying a malformed call. Track 3 problem #2 asks for
exactly this: *cut redundant, looping or failed tool calls without lowering the success rate.*
Tax filing is a good test bed because the answer is checkable to the dollar, and the documents
come with natural traps.

## The environment

Every case is generated from its id, so it is identical on every run and anyone can reproduce it.

- **Documents:** W-2s, 1099-INT / DIV / NEC with an expense ledger, Form 1098, donation receipts,
  plus decoys (an electric bill, last year's 1040, a bank statement, a health insurance card).
- **Tools:** `list_documents`, `read_document`, `calculator`, `lookup_standard_deduction`,
  `compute_tax` (2025 brackets), `submit_return`. Submitting returns no score, and only the first
  submission counts.
- **Traps:** a CORRECTED 1099-INT that replaces the original from the same bank; itemizing when
  mortgage interest plus donations beat the standard deduction; decoys.
- **Tiers:** easy (one W-2), medium (several income documents, one decoy), hard (2–3 decoys and
  traps), and **shift**: hard rules with employer, bank and person names, and decoy types (a W-4,
  a December pay stub, last year's 1099-INT), that never appear in training.

| Split | Cases | Purpose |
|---|---|---|
| train | 2,100 (700 per tier) | demonstrations only |
| eval | 300 (100 per tier) | main before/after |
| eval-shift | 150 | did it learn the skill or the surface? |

Splits are disjoint id ranges, and tests check that train-only decoys never leak into eval or shift.

## Metrics, and why these

- **Success:** all 11 return fields exact. The outcome that matters.
- **Field accuracy:** partial credit, so progress is visible before success moves.
- **Calls per case** and **wasted calls per case** (repeats, decoy reads, failed or malformed
  calls, anything after submitting).
- **Valid-call rate** (problem #1) and **AuditKIT precision vs. the oracle** (the share of the
  agent's calls that the minimal correct run would also make).
- **Tokens and seconds per case**, for the ROI comparison.

Success alone hides waste; waste alone rewards quitting early. We always report them together.

## Pipeline

```
TaxOps env ─► scripted oracle demos ─► SFT (Tiny Aya + LoRA) ─► eval traces ─► AuditKIT
                                     └► GRPO (AgentTune create_agentic_trainer, vLLM)  [attempt]
```

## Results

**Before → after on the same cases**

| Model | Cases | Success | Field acc. | Submitted | Calls | Wasted | AuditKIT precision |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global | eval, 60 | 0% | 9.7% | 25% | 38.5 | 27.8 | 0.27 |
| TaxOps SFT v3 | same 60 | **90.0%** | 96.1% | 100% | 10.9 | **0.03** | |
| Tiny Aya Global | shift, 22 | 0% | 5.8% | 18% | 38.4 | 24.7 | 0.37 |
| TaxOps SFT v3 | same 22 | **72.7%** | 90.9% | 100% | 14.9 | **0.23** | |
| Claude Sonnet | same 22 | 95.5% | 97.9% | 100% | 13.0 | 0.64 | |

**SFT v3 on the full held-out sets**

| Split | Success | Field acc. | Calls | Wasted | Valid calls | AuditKIT precision |
|---|---|---|---|---|---|---|
| eval (300) | 88.7% (easy 100 / medium 94 / hard 72) | 95.3% | 11.4 | 0.06 | 100% | 0.995 |
| eval-shift (150) | 68.7% | 86.9% | 15.2 | 0.47 | | 0.963 |


The base runs are smaller because the untrained model is slow (2–3 minutes a case on a free T4).
The base shift run was planned for 30 cases and ran out of GPU memory on case 23, so every model
is compared on the 22 it finished.

### What the base model does

It emits a whole batch of calls with guessed numbers before seeing a single result, then repeats
the batch every turn. It rarely calls `compute_tax`, invents values, and usually hits the 40-call
budget instead of submitting.

### Where SFT v3 still fails

| Cause | eval failures (34) | shift failures (47) |
|---|---|---|
| CORRECTED-1099 trap: read the original too and summed both | 18 | 26 |
| read a decoy and counted it | 0 | 9 |
| skipped a needed document | 4 | 4 |
| arithmetic or other | 12 | 8 |

The demonstrations never opened the superseded form, so the model never saw what to do once it
had. SFT v4 tried to fix that and got worse (next section); v6 fixed it a different way.

### SFT v4: a regression we couldn't pin down

v4 added 541 demonstrations that open the wrong document (the superseded 1099-INT, a decoy, a
missing document id, a bad filing status) and then carry on correctly. The wrong call was masked
out of the loss, so the model was never trained to make it, only to continue after it.

| | eval-300 v3 → v4 | shift-150 v3 → v4 |
|---|---|---|
| Success | 88.7% → 86.0% | 68.7% → **48.0%** |
| Wasted calls / case | 0.06 → 0.09 | 0.47 → 1.13 |
| Cases that opened a decoy | 0 → 4 | 45 → 97 |
| Cases that opened the superseded 1099 | 15 → 22 | 18 → 32 |
| Failures after opening a decoy | 0 → 3 | 10 → 38 |

v4 opened irrelevant documents far more often, and failed on most of the decoys it opened, worst
on decoy types it had never seen. We have two candidate causes and no run that separates them:
(1) wrong reads in context, even masked, may teach the model that opening extra documents is
normal; (2) v4 was also trained differently, with 4-bit QLoRA on two T4s instead of bf16 LoRA on an
A10G, then evaluated on the unquantized base, a train/eval mismatch v3 didn't have. Either could
explain the drop. v6 drops the wrong-read demos (it keeps only the bad-id and bad-status recoveries)
and trains in plain bf16/fp16 LoRA.

### SFT v6: read the CORRECTED form first

Instead of showing the model how to recover from reading the superseded form, v6 changes the
demonstrations so it never needs to: when a folder contains a CORRECTED 1099-INT, the oracle reads
it in a turn of its own, before anything else, and only then reads the rest, skipping the original
it replaces. Same number of tool calls, one extra turn. v6 continues training from the v3 adapter
on 600 new conversations (535 oracle, 65 bad-id/bad-status recoveries; trap-heavy hard cases,
cases with new train-only decoys, and replayed v3 cases), 1 epoch at lr 5e-5 on two T4s.

We picked it on a separate dev set (50 shifted cases outside every other split), with v3 and v6 in
the same Kaggle job on identical GPUs:

| dev-shift (50) | v3 | v6 |
|---|---|---|
| Success | 68% | **78%** |
| Field accuracy | 86.9% | 91.1% |
| Wasted calls / case | 0.42 | **0.14** |
| Calls over minimum (solved) | 0.24 | 0.08 |

<!-- FINAL: v3 vs v6 on eval-300 and eval-shift-150, same T4 hardware, from runs/kaggle-eval-final -->

On the full shift set, v6 opened the superseded 1099-INT in 9 cases (v3: 18) and a decoy in 25
(v3: 45). Case by case it fixed 20 of v3's failures and broke 12, a net gain that a sign test
can't yet separate from noise (p ≈ 0.22), so we call it consistent, not proven.

## The road there: what each version fixed

| Version | Change | eval-300 | shift-150 |
|---|---|---|---|
| v1 | oracle demos, no calculator | 40% on 30 cases: every miss a mental subtraction | |
| v2 | oracle does every subtraction with the calculator | 82.7% | 32% |
| v3 | also totals wages and interest with the calculator; train-only decoys | 88.7% | 68.7% |
| v4 | recovery demos for superseded and decoy reads (QLoRA, 2×T4) | 86.0% | 48.0% |

v2's 30-case sample said 96.7%. The full 300 said 82.7%. Always quote the full set.

## Closed-model comparison (ROI)

Each shift case went to a fresh Claude Code subagent (Sonnet) that could use only the TaxOps CLI,
from a folder without the source or answers. Across 30 shift cases: 96.7% success, 13.2 calls,
0.8 wasted, about 35k tokens per case. On the same 30, SFT v3 got 66.7% with 0.23 wasted.

Caveats: there was no API bill, so cost can only be estimated from token counts at public prices,
and latency isn't comparable with a self-hosted model. Transcripts are published.

## Findings (things that cost us time)

1. **Tiny Aya's chat template renders tool-role turns as empty.** AgentTune's distillation path
   puts tool results in `tool` messages, so the student never sees tool output. We wrote SFT in
   plain transformers + PEFT with tool results in user turns.
2. **EOS mismatch.** `generation_config` stops on `<EOS_TOKEN>`, but chat turns end with
   `<|END_OF_TURN_TOKEN|>`. Without stopping on it, generation runs on.
3. **Demonstrations must show the arithmetic.** A 3B model's mental sums are the main error source.
4. **GRPO needs reward variance.** From SFT, four rollouts at temperature 0.9 were identical
   (reward_std 0). Temperature 1.0 plus trap-only cases fixed it.
5. **GRPO + colocated vLLM with a 262k vocabulary** ran out of memory at batch 4 in backward;
   batch 1 × gradient accumulation 4 fits. vLLM cut a step from 544 s to 56 s.
6. **vLLM importance-sampling correction zeroed the loss**: every ratio was masked.
   `vllm_importance_sampling_correction=False` fixed it.
7. **Full logits don't fit on a T4.** A 3.5k-token conversation × 262k vocab is 3.4 GiB in fp32.
   Projecting only the trained positions (`logits_to_keep` with a boolean mask) brings peak memory
   to 8 GiB under 4-bit QLoRA.
8. **Overriding `Trainer.compute_loss` under DDP** skips the trainer's own multiply-by-world-size,
   so gradients come out halved. Easy to miss: the loss still looks plausible.

## What didn't work: GRPO

The plan was SFT to teach the workflow, then GRPO with a correctness-gated efficiency reward to
fix the traps. After fixing findings 4–6 it trained, but the GPU credit ran out at step 21 of 50.
Mean reward moved from 0.33 (first ten steps) to 0.37 (last ten), within noise, and no adapter
was saved. We report it as an attempt, not a result.

## Reproduce

See the repository README. Every number above is recomputed from the published traces by
`taxops.metrics` and `scripts/auditkit_report.py`.

## Credits

AgentTune and AuditKIT by Lexsi Labs; Tiny Aya by Cohere Labs. All data is synthetic.
