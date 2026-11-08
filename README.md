# TaxOps: a 3B agent that files tax returns without wasting tool calls

Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning). Problem statements
#2 (efficient trajectories) and #1 (reliable tool calling).

We fine-tune **Tiny Aya Global (3.35B)** to work through a folder of synthetic tax documents
with six tools and submit a correct federal return, while cutting redundant, failed and
decoy-reading calls. Pipeline: **TaxOps env → AgentTune (SFT, GRPO attempt) → AuditKIT**.

## Results

Before → after on the **same case ids** (base runs are smaller because the untrained model is slow):

| Model | Cases | Success | Field acc. | Submitted | Calls/case | Wasted/case | AuditKIT precision |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global (base) | eval, 60 | 0% | 9.7% | 25% | 38.5 | 27.8 | 0.27 |
| TaxOps SFT v3 | same 60 | 90.0% | 96.1% | 100% | 10.9 | 0.03 | — |
| **TaxOps SFT v6** (submitted) | same 60 | **95.0%** | 97.9% | 100% | 10.9 | **0.02** | — |
| Tiny Aya Global (base) | shift, 22 | 0% | 5.8% | 18% | 38.4 | 24.7 | 0.37 |
| TaxOps SFT v3 | same 22 | 72.7% | 90.9% | 100% | 14.9 | 0.23 | — |
| **TaxOps SFT v6** (submitted) | same 22 | **77.3%** | 92.1% | 100% | 14.9 | **0.09** | — |
| Claude Sonnet (closed) | same 22 | 95.5% | 97.9% | 100% | 13.0 | 0.64 | — |

Full held-out sets, v3 and v6 on the same hardware (Kaggle T4, fp16, greedy):

| Split | Model | Success | Field acc. | Calls/case | Wasted/case | Valid calls | AuditKIT precision |
|---|---|---|---|---|---|---|---|
| eval (300) | SFT v3 | 88.7% (easy 100, medium 94, hard 72) | 95.3% | 11.4 | 0.057 | 100% | 0.995 |
| eval (300) | **SFT v6** | **93.3%** (easy 100, medium 94, hard 86) | 97.1% | 11.4 | **0.030** | 100% | 0.996 |
| eval-shift (150) | SFT v3 | 67.3% | 86.4% | 15.2 | 0.47 | 100% | 0.963 |
| eval-shift (150) | **SFT v6** | **74.0%** | 89.9% | 15.0 | **0.27** | 100% | 0.976 |

v6 vs v3 case by case: eval 21 fixed / 7 broken (sign test p = 0.013), shift 21 / 11 (p = 0.11),
both 42 / 18 (p = 0.003). v3 gives the same outcome on A10G/bf16 and T4/fp16 on 300/300 eval and
148/150 shift cases.

**SFT v6** continues from v3 with demonstrations that read a CORRECTED 1099-INT first, in a turn
of its own, and skip the original it replaces. Cases that open the superseded form: eval 15 → 7,
shift 19 → 9. Chosen on a separate 50-case dev set (68% → 78%), never on eval or eval-shift.

SFT v4 (recovery demos: open the wrong document, then carry on, with the mistake masked from the
loss; QLoRA on 2×T4) was **worse**: eval 86.0%, shift 48.0%, and on shift it opened a decoy in
97 of 150 cases vs 45 for v3. Two candidate causes (the wrong-read demos, or 4-bit training vs
full-precision eval); we couldn't afford a run that separates them.

Base runs were done on a free Kaggle T4. The base shift run planned 30 cases but ran out of GPU
memory on case 23 (the base model's context grows to ~40 calls), so we report the 22 it finished
and compare every model on those same 22. Claude on all 30 shift cases: 96.7% success, 0.8 wasted (v6 on the same 30: 73.3%, 0.13).

Takeaway: SFT turns a model that never finishes into one that solves most cases at near the
minimum call count. Claude is more accurate on unseen decoys; the 3B model wastes about six times fewer calls.

## Environment

- Deterministic cases from a case id (easy / medium / hard / shift tiers). Fixed id ranges keep
  train (0–2099, plus 700+ / 20000+ ids for v6), eval (100000+), shift (200000+) and the dev
  sets used to pick models (hard 110000–110029, shift 300000–300049) disjoint; a test checks it.
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
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # CPU: env, tests, scoring
pip install -r requirements-gpu.txt                                 # GPU box: training and eval
PYTHONPATH=. .venv/bin/pytest -q tests         # env, verifier, rewards, split disjointness
python scripts/demo_scripted.py                # oracle / wasteful / careless agents

# SFT v3 (A10G, ~66 min). v4 later changed the demo mix and the train decoys, so v3's exact data
# lives at commit 196c8aa:
git checkout 196c8aa && python scripts/train_sft.py --out runs/sft-v3 --push <user>/taxops-tinyaya-sft-v3
# SFT v6: continue from v3 on the v6 mix (2×T4, ~2.8 h); kaggle/train/run.py runs exactly this
torchrun --nproc_per_node 2 scripts/train_sft.py --mix v6 --init-adapter <user>/taxops-tinyaya-sft-v3 \
  --lr 5e-5 --out runs/sft-v6

# Eval (any 16 GB GPU; T4 works in fp16). --shard i/n splits a split across GPUs.
python scripts/eval_model.py --adapter <user>/taxops-tinyaya-sft-v6 --split eval --out runs/sft-v6-eval
CUDA_VISIBLE_DEVICES=0 python scripts/eval_model.py --adapter ... --split eval-shift --shard 0/2 --out runs/s0 &
CUDA_VISIBLE_DEVICES=1 python scripts/eval_model.py --adapter ... --split eval-shift --shard 1/2 --out runs/s1
PYTHONPATH=. python scripts/merge_shards.py --out runs/sft-v6-shift runs/s0 runs/s1
python scripts/auditkit_report.py runs/sft-v6-shift/traces.jsonl
# Model selection used --split dev-shift (50 cases), never eval or eval-shift.
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

## Licence

Code: MIT (see `LICENSE`). The adapters are derivatives of Tiny Aya Global and follow its
CC-BY-NC-4.0 licence and Cohere Labs' Acceptable Use Policy. AgentTune and AuditKIT keep their own
licences.
