"""Stage 1: supervised warm start (LoRA) on oracle + recovery conversations.

Uses the exact message format of the agent loop and of AgentTune's Tiny Aya tool
fallback. Loss is on the model's own turns only; tool results, the prompt and the
deliberate mistakes in recovery examples are masked out.

Why not AgentTune's create_distill_trainer: its SFT rows put tool output in
"tool"-role turns, and Tiny Aya's chat template renders those as empty turns, so
the student would never see the documents (see README, "Findings").

    python scripts/train_sft.py --smoke                       # 2 steps, checks the setup
    python scripts/train_sft.py --limit 600 --out runs/sft    # T4-sized run
    python scripts/train_sft.py --push Manoj2142/taxops-tinyaya-sft
"""

import argparse
from collections import Counter
import json
import os
import time
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,  # noqa: E402
                          Trainer, TrainerCallback, TrainingArguments)

from taxops.data import SPLITS, sft_conversations  # noqa: E402


def encode(tok, row, max_len):
    """Tokenize one conversation; label only the assistant turns we want to learn."""
    msgs = row["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    spans, k = [], 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if k not in row.get("skip_assistant_turns", []):
            start = len(tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True))
            end = len(tok.apply_chat_template(msgs[:i + 1], tokenize=False))
            spans.append((start, end))
        k += 1
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    labels = [tid if any(s <= a < e for s, e in spans) else -100
              for tid, (a, _) in zip(enc["input_ids"], enc["offset_mapping"], strict=True)]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


class TrainedLogitsTrainer(Trainer):
    """Project onto the 262k-token vocab only where a label is trained (~16% of positions).
    Full logits for a 3.5k-token conversation need 3.4 GiB in fp32, more than a T4 has left."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        target = labels[:, 1:]
        keep = torch.zeros_like(labels, dtype=torch.bool)
        keep[:, :-1] = target != -100
        out = model(**inputs, logits_to_keep=keep)  # logits come back flattened to (n_kept, vocab)
        loss = torch.nn.functional.cross_entropy(out.logits.float(), target[target != -100].to(out.logits.device),
                                                 reduction="sum")
        loss = loss / (keep.sum() if num_items_in_batch is None else num_items_in_batch)
        if num_items_in_batch is not None and self.args.average_tokens_across_devices:
            # num_items_in_batch counts every GPU's tokens and DDP averages the gradients, so
            # scale back up, as Trainer's own compute_loss does.
            loss = loss * self.accelerator.num_processes
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--limit", type=int, default=None, help="train conversations (default: all 2,100)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--push", default=None, help="HF repo id for the adapter (created private)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly (and still save) after this long")
    ap.add_argument("--qlora", action="store_true",
                    help="load the base in 4-bit (for a 16 GB T4, which also has no bf16)")
    args = ap.parse_args()

    ids = list(SPLITS["train"])
    random.Random(0).shuffle(ids)  # mix tiers
    ids = ids[:200] if args.smoke else ids[: args.limit] if args.limit else ids
    rows = sft_conversations(ids)
    tok = AutoTokenizer.from_pretrained(args.model)
    data = [encode(tok, r, args.max_len) for r in rows]
    if args.smoke:  # the 2 longest of 200, so the smoke test also checks peak memory
        data = sorted(data, key=lambda d: -len(d["input_ids"]))[:2]
    trained = sum(sum(x != -100 for x in d["labels"]) for d in data)
    print(f"{len(data)} conversations, {sum(len(d['input_ids']) for d in data):,} tokens, {trained:,} trained tokens")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = None
    if args.qlora:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_use_double_quant=True)
    # QLoRA: the whole 4-bit model on this process's GPU (one per process under torchrun on
    # Kaggle's 2xT4). A model split across GPUs puts the hidden states on cuda:1 while the
    # trained-position mask sits on cuda:0.
    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 device_map={"": rank} if args.qlora else "auto",
                                                 quantization_config=quant)
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    if not bf16:
        # fp16 mixed precision needs fp32 master weights, or the grad scaler refuses to unscale.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    accum = 1 if args.smoke else max(1, 8 // world)  # effective batch 8 on any GPU count
    total_steps = 2 if args.smoke else max(1, int(len(data) * args.epochs / (accum * world)))
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1, gradient_accumulation_steps=accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, total_steps // 20),
        logging_steps=1 if args.smoke else 10, save_strategy="no", report_to="none",
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False,
    )
    callbacks = []
    if args.max_minutes:
        start = time.time()

        class TimeLimit(TrainerCallback):
            """Stop between steps once over budget, so the adapter still gets saved."""

            def on_step_end(self, a, state, control, **kw):
                if time.time() - start > args.max_minutes * 60:
                    control.should_training_stop = True

        callbacks.append(TimeLimit())
    trainer = TrainedLogitsTrainer(model=model, args=targs, train_dataset=data,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                                   callbacks=callbacks)
    trainer.train()
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if not trainer.is_world_process_zero():
        return
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "train_info.json").write_text(json.dumps({
        "base_model": args.model, "conversations": len(data), "epochs": args.epochs, "lr": args.lr,
        "qlora": args.qlora, "gpus": world, "steps_done": trainer.state.global_step,
        "steps_planned": trainer.state.max_steps, "kinds": dict(Counter(r["kind"] for r in rows)),
        "log_history": trainer.state.log_history}, indent=2))
    print("saved adapter to", args.out)
    if args.push and not args.smoke:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.push, folder_path=args.out, commit_message="SFT adapter")
        print("pushed to", args.push, "(private)")


if __name__ == "__main__":
    main()
