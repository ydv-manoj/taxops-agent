---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
tags: [lora, agents, tool-calling, sft, lexsi-hackathon]
datasets: [Manoj2142/taxops-traces]
---

# TaxOps SFT v6 (LoRA for Tiny Aya Global)

A LoRA adapter that teaches **Tiny Aya Global (3.35B)** to file a synthetic federal tax return in
the [TaxOps](https://github.com/ydv-manoj/taxops-agent) environment with six tools, using close to
the minimum number of calls. Built for the Lexsi AI Safety & Alignment Hackathon, Track 3
(efficient trajectories, reliable tool calling).

## Results (greedy decoding, T4, fp16)

| Split | Model | Success | Field acc. | Wasted calls / case | Valid calls |
|---|---|---|---|---|---|
| eval (60 of 300) | Tiny Aya Global, untrained | 0% | 9.7% | 27.8 | |
| eval (300) | SFT v3 | 88.7% | 95.3% | 0.057 | 100% |
| eval (300) | **SFT v6** | **93.3%** (hard tier 86%) | 97.1% | **0.030** | 100% |
| eval-shift (150) | SFT v3 | 67.3% | 86.4% | 0.47 | 100% |
| eval-shift (150) | **SFT v6** | **74.0%** | 89.9% | **0.27** | 100% |

Success means all 11 return fields exactly right. "Shift" cases use employer, bank and person
names and decoy document types never seen in training. Every number is recomputed from the
published traces.

## How it was trained

- Continued from [SFT v3](https://huggingface.co/Manoj2142/taxops-tinyaya-sft-v3) (itself trained on
  2,100 scripted demonstrations).
- 600 new conversations: 535 oracle runs (trap-heavy hard cases, cases with new train-only
  decoys, replayed v3 cases) and 65 recoveries from a bad document id or filing status. When a
  folder has a CORRECTED 1099-INT, the demonstration reads it first, in a turn of its own.
- LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections; 1 epoch, lr 5e-5,
  fp16 on 2× T4 (DDP), loss on assistant turns only. About 2.8 hours.
- Checked on a separate 50-case shifted dev set before the full runs (v3 68% → v6 78%). Its data
  was designed from v3's failures on eval and eval-shift, so those sets are not untouched for v6.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "Manoj2142/taxops-tinyaya-sft-v6").merge_and_unload()
```

The model expects the TaxOps prompt and tool format; run it through `taxops.agent_loop.run_episode`
(see `scripts/eval_model.py`). Stop generation on `<|END_OF_TURN_TOKEN|>` and `<|END_RESPONSE|>`
as well as `<EOS_TOKEN>`.

## Limits

- Synthetic data and a simplified 2025 federal subset (wages, interest, dividends, simple
  Schedule C, standard or itemized deduction). Not tax advice and not for real filings.
- On the shifted set it still fails about a quarter of cases, spread across reading a decoy or
  the superseded form, skipping a needed document, and arithmetic or other mistakes.
- Against v3, case by case: eval 21 fixed / 7 broken (sign test p = 0.013); shift 21 / 11
  (p = 0.11, not significant on its own).

## Licence

CC-BY-NC-4.0, following the base model; use must also follow
[Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).
