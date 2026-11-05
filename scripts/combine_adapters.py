"""Combine LoRA adapters in weight space, exactly, without a GPU.

Each adapter adds s_i * B_i @ A_i to a base weight (s_i = lora_alpha / r). A weighted sum
sum_i w_i * s_i * B_i @ A_i is itself a LoRA of rank sum(r_i): stack the A's, and stack the B's
scaled by w_i * s_i, with lora_alpha = r so the new scaling is 1. Weights may be negative,
so "v3 - 0.5 * (v4 - v3)" is --adapters v3 v4 --weights 1.5 -0.5.

    python scripts/combine_adapters.py --adapters Manoj2142/taxops-tinyaya-sft-v3 runs/.../sft-v4 \
        --weights 1.5 -0.5 --out runs/combo-v3x
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file, save_file


def local(path: str) -> Path:
    p = Path(path)
    return p if p.exists() else Path(snapshot_download(path, allow_patterns=["adapter_*", "tokenizer*", "chat_template*"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.adapters) == len(args.weights)

    dirs = [local(a) for a in args.adapters]
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    tensors = [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    assert all(set(t) == set(tensors[0]) for t in tensors), "adapters target different modules"
    assert not any(c.get("use_rslora") or c.get("use_dora") for c in cfgs)

    out = {}
    for key in tensors[0]:
        if ".lora_A." in key:
            out[key] = np.concatenate([t[key] for t in tensors], axis=0)
        elif ".lora_B." in key:
            out[key] = np.concatenate([w * c["lora_alpha"] / c["r"] * t[key]
                                       for w, c, t in zip(args.weights, cfgs, tensors)], axis=1)
        else:
            raise ValueError(f"unexpected tensor {key}")

    rank = sum(c["r"] for c in cfgs)
    cfg = dict(cfgs[0], r=rank, lora_alpha=rank, lora_dropout=0.0)
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    save_file(out, str(o / "adapter_model.safetensors"))
    (o / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for f in dirs[0].iterdir():
        if f.name.startswith(("tokenizer", "chat_template")):
            shutil.copy(f, o / f.name)
    (o / "combine_info.json").write_text(json.dumps({"adapters": args.adapters, "weights": args.weights}, indent=2))
    print(f"wrote rank-{rank} adapter to {o}")


if __name__ == "__main__":
    main()
