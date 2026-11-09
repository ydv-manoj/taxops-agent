# TaxOps: teaching a 3B model to file a tax return without wasting tool calls

## TL;DR

I fine-tuned **Tiny Aya Global (3.35B)** to work through a folder of synthetic tax documents
with six tools and file a federal return. The untrained model never files a correct return and
spends about 28 of its ~38 calls on waste. After LoRA fine-tuning on scripted
demonstrations (scored with the TaxOps verifier and AuditKIT), it files **93.3%** of 300 held-out
returns exactly right with **0.03 wasted calls per case**, and **74.0%** on a harder shifted set it
never saw. The biggest single gain came from changing *what the demonstrations do* around a trap,
not from adding examples of recovering from it. Claude Sonnet, given the same tools through a CLI, is
more accurate on 30 shifted cases (96.7% vs 73.3%) but wastes about six times as many calls on
them (0.8 vs 0.13). I also report what didn't work: a GRPO stage (AgentTune) that ran out of GPU
credit at step 21, and a handful of tooling bugs that may save you some time.

**Links:** [code](https://github.com/ydv-manoj/taxops-agent) ·
[model (SFT v6)](https://huggingface.co/Manoj2142/taxops-tinyaya-sft-v6) ·
[data and traces](https://huggingface.co/datasets/Manoj2142/taxops-traces) ·
[replay demo](https://huggingface.co/spaces/Manoj2142/taxops-replay) ·
[models collection](https://huggingface.co/collections/Manoj2142/taxops-models) ·
[data collection](https://huggingface.co/collections/Manoj2142/taxops-data-and-traces)

## Why this problem

In long-running agent workflows, much of the cost is tool calls and the tokens they drag into
the context: re-reading a document, opening a file that doesn't matter, retrying a malformed call. Track 3 problem #2 asks for
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
| dev, dev-shift | 30 hard, 50 shifted | choosing between model versions; never eval or eval-shift |

v6 also trains on 600 conversations from further train-only ids (see below).

Splits are disjoint id ranges, and tests check that train-only decoys never leak into eval or shift.

## Metrics, and why these

- **Success:** all 11 return fields exact. The outcome that matters.
- **Field accuracy:** partial credit, so progress is visible before success moves.
- **Calls per case** and **wasted calls per case** (repeats, decoy reads, failed or malformed
  calls, anything after submitting).
- **Valid-call rate** (problem #1) and **AuditKIT precision vs. the oracle** (AuditKIT's
  name-level tool-call precision against the scripted oracle's calls for the same case).
- **Tokens and seconds per case**, for the cost comparison.

Success alone hides waste; waste alone rewards quitting early. I always report them together.

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
| TaxOps SFT v3 | same 60 | 90.0% | 96.1% | 100% | 10.9 | 0.03 | |
| **TaxOps SFT v6** | same 60 | **95.0%** | 97.9% | 100% | 10.9 | **0.02** | |
| Tiny Aya Global | shift, 22 | 0% | 5.8% | 18% | 38.4 | 24.7 | 0.37 |
| TaxOps SFT v3 | same 22 | 72.7% | 90.9% | 100% | 14.9 | 0.23 | |
| **TaxOps SFT v6** | same 22 | **77.3%** | 92.1% | 100% | 14.9 | **0.09** | |
| Claude Sonnet | same 22 | 95.5% | 97.9% | 100% | 13.0 | 0.64 | |

**Full held-out sets, v3 and v6 on the same hardware** (Kaggle T4, fp16, greedy)

| Split | Model | Success | Field acc. | Calls | Wasted | Valid calls | AuditKIT precision |
|---|---|---|---|---|---|---|---|
| eval (300) | SFT v3 | 88.7% (easy 100 / medium 94 / hard 72) | 95.3% | 11.4 | 0.057 | 100% | 0.995 |
| eval (300) | **SFT v6** | **93.3%** (easy 100 / medium 94 / hard **86**) | 97.1% | 11.4 | **0.030** | 100% | 0.996 |
| eval-shift (150) | SFT v3 | 67.3% | 86.4% | 15.2 | 0.47 | 100% | 0.963 |
| eval-shift (150) | **SFT v6** | **74.0%** | 89.9% | 15.0 | **0.27** | 100% | 0.976 |

Case by case, v6 fixes 21 of v3's eval failures and breaks 7 (sign test p = 0.013); on shift it
fixes 21 and breaks 11 (p = 0.11); both sets together, 42 vs 18 (p = 0.003). Earlier v3 numbers
came from an A10G in bf16; rerunning v3 on the T4 gave the same outcome on 300/300 eval and 148/150
shift cases, so hardware isn't what separates the two models.


The base runs are smaller because the untrained model is slow (2–3 minutes a case on a free T4).
The base shift run was planned for 30 cases and ran out of GPU memory on case 23, so every model
is compared on the 22 it finished.

### What the base model does

It emits a whole batch of calls with guessed numbers before seeing a single result, then repeats
the batch every turn. It never called `compute_tax` in any of the 82 base runs, invents values, and in 41
of 60 eval cases hit the 40-call budget instead of submitting.

### Where SFT v3 still fails

| Cause (first that applies) | eval failures (34) | shift failures (47) |
|---|---|---|
| opened the superseded 1099-INT (CORRECTED-form trap) | 15 | 18 |
| opened a decoy | 0 | 10 |
| skipped a needed document | 5 | 8 |
| arithmetic or other | 14 | 11 |

The demonstrations never opened the superseded form, so the model never saw what to do once it
had. SFT v4 tried to fix that and got worse (next section); v6 fixed it a different way.

### SFT v4: a regression I couldn't pin down

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
on decoy types it had never seen. I have two candidate causes and no run that separates them:
(1) wrong reads in context, even masked, may teach the model that opening extra documents is
normal; (2) v4 was also trained differently, with 4-bit QLoRA on two T4s instead of bf16 LoRA on an
A10G, then evaluated on the unquantized base, a train/eval mismatch v3 didn't have. Either could
explain the drop. v6 drops the wrong-read demos (it keeps only the bad-id and bad-status recoveries)
and trains in fp16 LoRA without quantization.

### SFT v6: read the CORRECTED form first

Instead of showing the model how to recover from reading the superseded form, v6 changes the
demonstrations so it never needs to: when a folder contains a CORRECTED 1099-INT, the oracle reads
it in a turn of its own, before anything else, and only then reads the rest, skipping the original
it replaces. Same number of tool calls, one extra turn. v6 continues training from the v3 adapter
on 600 new conversations (535 oracle, 65 bad-id/bad-status recoveries; trap-heavy hard cases,
cases with new train-only decoys, and replayed v3 cases), 1 epoch at lr 5e-5 on two T4s.

I picked it on a separate dev set (50 shifted cases outside every other split), with v3 and v6 in
the same Kaggle job on identical GPUs:

| dev-shift (50) | v3 | v6 |
|---|---|---|
| Success | 68% | **78%** |
| Field accuracy | 86.9% | 91.1% |
| Wasted calls / case | 0.42 | **0.14** |
| Calls over minimum (solved) | 0.24 | 0.08 |

On the held-out sets (same hardware, full table above):

| | eval-300 v3 → v6 | shift-150 v3 → v6 |
|---|---|---|
| Success | 88.7% → **93.3%** | 67.3% → **74.0%** |
| Hard tier | 72% → **86%** | |
| Cases that opened the superseded 1099 | 15 → **7** | 19 → **9** |
| Cases that opened a decoy | 0 → 0 | 45 → **25** |
| Failures: superseded form / decoy / skipped a document / other | 15 / 0 / 5 / 14 → **7 / 0 / 5 / 8** | 19 / 11 / 8 / 11 → **9 / 8 / 9 / 13** |

The trap it targeted is half as common, and nothing else got worse on eval. On shift, the
remaining failures are spread across decoys the model has never seen, skipped documents, and
arithmetic or other mistakes. That's where I'd go next, with on-policy data (sample the model, keep its correct
runs, fix its wrong ones) rather than more scripted demonstrations.

## The road there: what each version fixed

| Version | Change | eval-300 | shift-150 |
|---|---|---|---|
| v1 | oracle demos, no calculator | 40% on 30 cases: every miss a mental subtraction | |
| v2 | oracle does every subtraction with the calculator | 82.7% | 32% |
| v3 | also totals wages and interest with the calculator; train-only decoys | 88.7% | 68.7% (67.3% on T4) |
| v4 | recovery demos for superseded and decoy reads (QLoRA, 2×T4) | 86.0% | 48.0% |
| **v6** | continue v3; read the CORRECTED form first; no wrong-read demos (fp16, 2×T4) | **93.3%** | **74.0%** |

v2's 30-case sample said 96.7%. The full 300 said 82.7%. Always quote the full set.

## Closed-model comparison

Each shift case went to a fresh Claude Code subagent (Sonnet) that could use only the TaxOps CLI,
from a folder without the source or answers. Across 30 shift cases: 96.7% success, 13.2 calls,
0.8 wasted. On the same 30, SFT v3 got 66.7% with 0.23 wasted and SFT v6 73.3% with 0.13 wasted.

Caveats: there was no API bill, so I don't quote a dollar cost. Each Claude subagent used about
35k tokens, but that includes Claude Code's own instructions, so it isn't comparable with the
~18k prompt tokens per case the 3B model reads. Latency isn't comparable either. Claude's tool
calls and CLI logs are published with the traces.

## Findings (things that cost me time)

1. **Tiny Aya's chat template renders tool-role turns as empty.** AgentTune's distillation path
   puts tool results in `tool` messages, so the student never sees tool output. I wrote SFT in
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
was saved. I report it as an attempt, not a result.

## Reproduce

See the [repository README](https://github.com/ydv-manoj/taxops-agent#reproduce). Every number above is recomputed from the published traces by
`taxops.metrics` and `scripts/auditkit_report.py`.

## Credits

AgentTune and AuditKIT by Lexsi Labs; Tiny Aya by Cohere Labs. All data is synthetic.
