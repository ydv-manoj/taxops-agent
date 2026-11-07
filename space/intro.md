# TaxOps replay

Recorded runs of agents filing a synthetic 2025 federal tax return with six tools. Pick a case to
see each agent's tool calls side by side; calls the verifier counts as **wasted** (repeats,
irrelevant or superseded documents, failed calls, anything after submitting) are flagged ⚠️.

- **Tiny Aya Global (3.35B), untrained**, then after LoRA fine-tuning (**SFT v3**, **SFT v6**).
- **Claude Sonnet**, given the same tools through a CLI, as a closed-model reference.
- **eval-shift** cases use employer, bank and person names and decoy types never seen in training.

Nothing runs a model here: every run is a recorded trace, rescored live with the same verifier
that produced the published numbers. The untrained model and Claude were run on fewer cases
(their columns say so when a case is missing).
[Code](https://github.com/ydv-manoj/taxops-agent) ·
[Traces](https://huggingface.co/datasets/Manoj2142/taxops-traces) ·
[Model](https://huggingface.co/Manoj2142/taxops-tinyaya-sft-v6)
