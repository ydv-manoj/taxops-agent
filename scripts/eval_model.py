"""Evaluate a model (optionally with a LoRA adapter) on a TaxOps split. Needs a GPU.

    python scripts/eval_model.py --out runs/base --limit 60
    python scripts/eval_model.py --adapter Manoj2142/taxops-sft --out runs/sft

Writes <out>/traces.jsonl (every episode, scored) and <out>/summary.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from taxops.agent_loop import parse_calls, run_episode, tool_schemas  # noqa: E402
from taxops.data import SPLITS  # noqa: E402
from taxops.metrics import score_episode, summarize  # noqa: E402


def pick_ids(split: str, limit: int | None) -> list[str]:
    ids = SPLITS[split]
    if not limit:
        return ids
    tiers = sorted({i.rsplit("-", 1)[0] for i in ids})
    per = max(1, limit // len(tiers))
    return [i for t in tiers for i in [x for x in ids if x.startswith(t + "-")][:per]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="eval", choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="cases in total, spread evenly across tiers")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th case starting at i (one GPU per shard)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        repo, sub = args.adapter, None
        if not Path(repo).exists() and repo.count("/") > 1:  # "user/repo/subfolder" on the Hub
            user, name, sub = repo.split("/", 2)
            repo = f"{user}/{name}"
        kw = {"subfolder": sub} if sub else {}
        model = PeftModel.from_pretrained(model, repo, **kw).merge_and_unload()
    model.eval()

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    # Tiny Aya's generation_config stops on <EOS_TOKEN>, but chat turns end with
    # <|END_RESPONSE|><|END_OF_TURN_TOKEN|>; stop on all of them.
    stop_ids = sorted({i for i in (tok.convert_tokens_to_ids(t) for t in
                                   ("<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"))
                       if isinstance(i, int) and i != tok.unk_token_id} | {tok.eos_token_id})

    @torch.no_grad()
    def generate(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=stop_ids,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        usage["prompt_tokens"] += ids["input_ids"].shape[1]
        usage["completion_tokens"] += len(new)
        return tok.decode(new, skip_special_tokens=True).strip()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = tool_schemas()
    scores, turns, valid_turns, seconds = [], 0, 0, 0.0
    case_ids = pick_ids(args.split, args.limit)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        case_ids = case_ids[i::n]
    with open(out_dir / "traces.jsonl", "w") as f:
        for n, cid in enumerate(case_ids, 1):
            before = dict(usage)
            t0 = time.time()
            ep = run_episode(cid, generate, schemas)
            dt = time.time() - t0
            seconds += dt
            s = score_episode(cid, ep["calls"])
            scores.append(s)
            replies = [m["content"] for m in ep["messages"] if m["role"] == "assistant"]
            turns += len(replies)
            valid_turns += sum(parse_calls(r) is not None for r in replies)
            ep.update(score=s.to_dict(), seconds=round(dt, 2),
                      tokens={k: usage[k] - before[k] for k in usage})
            f.write(json.dumps(ep) + "\n")
            f.flush()
            print(f"[{n}/{len(case_ids)}] {cid}: success={s.success} fields={s.field_accuracy:.0%} "
                  f"calls={s.tool_calls} wasted={s.wasted_calls} stop={ep['stop']} {dt:.0f}s", flush=True)

    summary = {"model": args.model, "adapter": args.adapter, "split": args.split, "shard": args.shard,
               "overall": summarize(scores),
               "by_tier": {t: summarize([s for s in scores if s.case_id.startswith(t + "-")])
                           for t in sorted({s.case_id.rsplit("-", 1)[0] for s in scores})},
               "valid_call_rate": valid_turns / max(1, turns),
               "avg_seconds_per_case": seconds / max(1, len(scores)),
               "avg_tokens_per_case": {k: v / max(1, len(scores)) for k, v in usage.items()},
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2), "\nvalid_call_rate:", summary["valid_call_rate"])


if __name__ == "__main__":
    main()
