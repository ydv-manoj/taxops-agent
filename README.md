# TaxOps: a 3B agent that files tax returns without wasting tool calls

Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning). Problem statements
#2 (efficient trajectories) and #1 (reliable tool calling).

We fine-tune **Tiny Aya Global (3.35B)** to work through a folder of synthetic tax documents
with six tools and submit a correct federal return, while cutting redundant, failed and
decoy-reading calls. Pipeline: **TaxOps env → AgentTune (SFT, GRPO attempt) → AuditKIT**.

> DRAFT.

## Results

Before → after on the **same case ids** (base runs are smaller because the untrained model is slow):

| Model | Cases | Success | Field acc. | Submitted | Calls/case | Wasted/case | AuditKIT precision |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | eval, 60 | 0% | 9.7% | 25% | 38.5 | 27.8 | 0.27 |
| **TaxOps SFT v3** | same 60 | **90.0%** | 96.1% | 100% | 10.9 | 0.03 | — |
| Tiny Aya Global (base) | shift, 22 | 0% | 5.8% | 18% | 38.4 | 24.7 | 0.37 |
| **TaxOps SFT v3** | same 22 | **72.7%** | 90.9% | 100% | 14.9 | 0.23 | — |
| Claude Sonnet (closed) | same 22 | 95.5% | 97.9% | 100% | 13.0 | 0.64 | — |

Full held-out sets, SFT v3:

| Split | Success | Field acc. | Calls/case | Wasted/case | Valid calls | AuditKIT precision |
|---|---|---|---|---|---|---|
| eval (300) | 88.7% (easy 100, medium 94, hard 72) | 95.3% | 11.4 | 0.06 | 100% | 0.995 |
| eval-shift (150) | 68.7% | 86.9% | 15.2 | 0.47 | — | 0.963 |

Base runs were done on a free Kaggle T4. The base shift run planned 30 cases but ran out of GPU
memory on case 23 (the base model's context grows to ~40 calls), so we report the 22 it finished
and compare every model on those same 22. Claude on all 30 shift cases: 96.7% success, 0.8 wasted.

Takeaway: SFT turns a model that never finishes into one that solves most cases at near the
minimum call count. Claude is more accurate on unseen decoys; the 3B model wastes fewer calls.

## Environment

- Deterministic cases from a case id (easy / medium / hard / shift tiers). Fixed id ranges keep
  train (0–2099), eval (100000+) and shift (200000+) disjoint; a test checks it.
- Six stateless tools: `list_documents`, `read_document`, `calculator`,
  `lookup_standard_deduction`, `compute_tax`, `submit_return`.
- Traps: a CORRECTED 1099-INT that supersedes the original, itemize-vs-standard choice, decoy
  documents. The shift split uses unseen names and unseen decoys (W-4, pay stub, prior-year 1099-INT).
- Verifier: all 11 return fields must match. Wasted = repeats + decoy reads + failed calls +
  calls after submit.

## Metrics and why

Success alone hides waste; efficiency alone rewards giving up. We report both, plus field
accuracy (partial credit), valid-call rate (problem #1), and tokens / seconds per case (ROI).

## Findings

1. **Tiny Aya's chat template drops tool-role turns.** AgentTune's distill path renders tool
   results as role `tool`, which Tiny Aya's template renders empty, so the student never sees tool
   output. We wrote SFT with plain transformers + PEFT instead.
2. **EOS mismatch.** `generation_config` stops on `<EOS_TOKEN>`, but chat turns end with
   `<|END_OF_TURN_TOKEN|>`; without stopping on it the model keeps generating.
3. **Demonstrations must show the arithmetic.** SFT v1 (no calculator in the oracle) got 40%:
   every miss was a mental subtraction. v2 added calculator steps; v3 also totals multi-document
   fields (v2 slipped digits like 266,500 vs 266,600) and adds train-only decoys → shift 32% → 68.7%.
4. **GRPO needs variance.** From SFT, 4 rollouts at temperature 0.9 were identical (reward_std 0).
   Temperature 1.0 plus trap-only cases fixed it.
5. **GRPO + vLLM on a 262k vocab.** Batch-4 backward ran out of memory next to colocated vLLM;
   batch 1 × grad accumulation 4 fits. vLLM cut step time from 544 s to 56 s.
6. **vLLM importance-sampling correction zeroed the loss.** With the default, every IS ratio was
   masked and loss was 0; `vllm_importance_sampling_correction=False` fixed it.
7. **GRPO was cut short.** The run stopped at step 21/50 when the GPU credit ran out; reward went
   0.33 → 0.37 (noisy). We report it as an attempt, not a result.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # TODO: add requirements.txt
.venv/bin/pytest -q tests                      # env, verifier, rewards, split disjointness
python scripts/demo_scripted.py                # oracle / wasteful / careless agents

# SFT (A10G, ~66 min)
python scripts/train_sft.py --out runs/sft-v3 --push <user>/taxops-tinyaya-sft-v3
# Eval (any 16 GB+ GPU; T4 works in fp16)
python scripts/eval_model.py --adapter <user>/taxops-tinyaya-sft-v3 --split eval --out runs/sft-v3-eval
python scripts/eval_model.py --adapter <user>/taxops-tinyaya-sft-v3 --split eval-shift --out runs/sft-v3-shift
python scripts/auditkit_report.py runs/sft-v3-eval/traces.jsonl
# GRPO attempt (A100, AgentTune + vLLM)
python scripts/train_grpo.py --sft-adapter <user>/taxops-tinyaya-sft-v3 --only-notes corrected_1099 \
  --vllm --steps 50 --lr 2e-5 --temperature 1.0
```

Claude baseline: one fresh Claude Code subagent (Sonnet) per case, using only `taxops/cli.py`
from a folder without the source or answers. Cost is estimated from token counts; there was no
API bill, so latency is not comparable.

## Credits

AgentTune and AuditKIT (Lexsi Labs), Tiny Aya (Cohere Labs). All data is synthetic; no real or
employer data is used.
