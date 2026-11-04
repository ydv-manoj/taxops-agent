---
title: "TaxOps: teaching a 3B model to file a tax return without wasting tool calls"
authors: Manoj Kumar
---

<!-- DRAFT for the HF Article. SFT v4 numbers pending (training on Kaggle 2xT4). -->

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

<!-- TODO: SFT v4 rows -->

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
had. SFT v4 adds demonstrations that read the wrong document and then leave it out.

## The road there: what each version fixed

| Version | Change | eval-300 | shift-150 |
|---|---|---|---|
| v1 | oracle demos, no calculator | 40% on 30 cases: every miss a mental subtraction | |
| v2 | oracle does every subtraction with the calculator | 82.7% | 32% |
| v3 | also totals wages and interest with the calculator; train-only decoys | 88.7% | 68.7% |
| v4 | recovery demos for superseded and decoy reads | TODO | TODO |

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
