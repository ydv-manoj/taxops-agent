---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
tags: [lora, agents, tool-calling, sft, lexsi-hackathon]
datasets: [Manoj2142/taxops-traces]
---

# TaxOps SFT v3 (LoRA for Tiny Aya Global)

The first strong TaxOps adapter, and the starting point for the submitted model,
**[SFT v6](https://huggingface.co/Manoj2142/taxops-tinyaya-sft-v6)**. Use v6 unless you want to
reproduce the v3 → v6 comparison.

It teaches **Tiny Aya Global (3.35B)** to file a synthetic federal tax return in the
[TaxOps](https://github.com/ydv-manoj/taxops-agent) environment with six tools.

## Results (greedy decoding, T4, fp16)

| Split | Success | Field acc. | Wasted calls / case | Valid calls |
|---|---|---|---|---|
| eval (300) | 88.7% (easy 100, medium 94, hard 72) | 95.3% | 0.057 | 100% |
| eval-shift (150) | 67.3% | 86.4% | 0.47 | 100% |

The untrained base model solves 0% of the same eval cases. An earlier run on an A10G in bf16
gave the same outcome on 300/300 eval and 148/150 shift cases.

## How it was trained

- 2,100 scripted oracle demonstrations (700 easy, medium and hard cases), plus recoveries from a
  bad document id or filing status, with the wrong call masked out of the loss.
- The demonstrations do every sum and subtraction with the calculator tool; train-only decoy
  documents, distinct from the shifted test set's decoys.
- LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections; 1 epoch, lr 1e-4,
  bf16 on one A10G, loss on assistant turns only. Data at repository commit `196c8aa`.

## Use

Same as v6: load the base, apply the adapter with PEFT, and run it through
`taxops.agent_loop.run_episode` (see `scripts/eval_model.py`).

## Limits

Synthetic data and a simplified 2025 federal subset; not tax advice. Its most common failure is
the CORRECTED 1099-INT trap (reading the superseded form as well), which v6 halves.

## Licence

CC-BY-NC-4.0, following the base model; use must also follow
[Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).
