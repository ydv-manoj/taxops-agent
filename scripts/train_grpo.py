"""Stage 2: GRPO with AgentTune on live TaxOps rollouts.

Starts from the SFT adapter (merged into the base model), runs the six tools for
real during every rollout, and scores each rollout with taxops.rewards.taxops_reward.

    python scripts/train_grpo.py --sft-adapter runs/sft --smoke
    python scripts/train_grpo.py --sft-adapter Manoj2142/taxops-tinyaya-sft --steps 150 --push Manoj2142/taxops-tinyaya-grpo
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from taxops import env  # noqa: E402
from taxops.agent_loop import MAX_STEPS, RESULT_FORMAT, TOOLS_PROMPT  # noqa: E402
from taxops.data import SPLITS  # noqa: E402
from taxops.rewards import taxops_reward  # noqa: E402

STOP_TOKENS = ("<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>")


def prepare_start_model(base: str, sft_adapter: str | None, out: Path) -> str:
    """Merge the SFT adapter (if any) into the base model and save it locally, with a
    generation config that stops at the end of a chat turn (Tiny Aya's own config
    only stops at <EOS_TOKEN>, which chat turns never emit)."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    if sft_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_adapter).merge_and_unload()
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    model.save_pretrained(out)
    tok.save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--generations", type=int, default=4, help="rollouts per case (the GRPO group)")
    ap.add_argument("--tiers", default="medium,hard", help="train tiers; easy is solved by SFT already")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--push", default=None)
    args = ap.parse_args()

    from agenttune.core.backend_factory import create_agentic_trainer

    out = Path(args.out)
    start = prepare_start_model(args.model, args.sft_adapter, out / "start-model")

    tiers = tuple(args.tiers.split(","))
    ids = [c for c in SPLITS["train"] if c.startswith(tiers)]
    random.Random(1).shuffle(ids)
    ids = ids[:2] if args.smoke else ids
    train = Dataset.from_list([{"prompt": [{"role": "user", "content": env.task_prompt(c)}], "case_id": c}
                               for c in ids])

    gen = 2 if args.smoke else args.generations
    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=start,
        train_dataset=train,
        tools=env.TOOLS,
        reward_funcs=[taxops_reward],
        tools_fallback_prompt=TOOLS_PROMPT,   # same system prompt as our eval loop
        tool_result_format=RESULT_FORMAT,
        max_steps_per_turn=MAX_STEPS,
        output_dir=str(out),
        max_steps=2 if args.smoke else args.steps,
        per_device_train_batch_size=gen,
        gradient_accumulation_steps=1,
        num_generations=gen,
        max_completion_length=512,
        learning_rate=args.lr,
        temperature=0.9,
        beta=0.02,
        logging_steps=1,
        save_steps=50,
        use_vllm=False,
        peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        report_to="none",
    )
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    log = [h for h in trainer.trainer.state.log_history if "reward" in h]
    (out / "train_info.json").write_text(json.dumps({"start_model": args.sft_adapter or args.model,
                                                     "cases": len(ids), "log_history": log}, indent=2))
    if log:
        print(f"reward {log[0]['reward']:.3f} -> {log[-1]['reward']:.3f} over {len(log)} logged steps")
    if args.push and not args.smoke:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.push, folder_path=str(out), commit_message="GRPO adapter + trajectories",
                          ignore_patterns=["start-model/**", "checkpoint-*/**"])
        print("pushed to", args.push, "(private)")


if __name__ == "__main__":
    main()
