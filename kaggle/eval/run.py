# TaxOps eval on a Kaggle GPU. Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# Earlier runs: untrained Tiny Aya (eval --limit 60, eval-shift --limit 30), SFT v3/v4 (full splits; v3 on
# an HF A10G in bf16), SFT v6 on eval-shift. This run puts every final number on the same hardware
# (T4, fp16): v6 on eval, and v3 on eval + eval-shift. Each job is split in two (every 2nd case), and
# GPU 0 and GPU 1 each work through their own queue of shards; per-case lines go to the main log.
V3, V6 = "Manoj2142/taxops-tinyaya-sft-v3", "Manoj2142/taxops-tinyaya-sft-v6"
E = "python scripts/eval_model.py"
W = "/kaggle/working"
def job(gpu, adapter, name, split, shard):
    out = f"{name}-{split}-{shard[0]}"
    return (f"CUDA_VISIBLE_DEVICES={gpu} {E} --adapter {adapter} --split {split} --shard {shard} "
            f"--out {W}/runs/{out} 2>&1 | tee {W}/{out}.log | grep --line-buffered -E '^\\[|success_rate' | sed -u 's/^/{out} /'")
Q0 = " ; ".join([job(0, V6, "v6", "eval", "0/2"), job(0, V3, "v3", "eval", "0/2"), job(0, V3, "v3", "eval-shift", "0/2")])
Q1 = " ; ".join([job(1, V6, "v6", "eval", "1/2"), job(1, V3, "v3", "eval", "1/2"), job(1, V3, "v3", "eval-shift", "1/2")])
RUN = f"({Q0}) & P=$!; ({Q1}); wait $P"


# Read-only HF token from the private dataset papw123/taxops-hf-token (CLI-pushed runs
# can't read Kaggle Secrets). Falls back to the secret for runs started in the editor.
found = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
if found:
    os.environ["HF_TOKEN"] = open(found[0]).read().strip()
else:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft", "accelerate",
                "huggingface_hub"], check=True)
# Kaggle ships torchao 0.10; new peft refuses it when wrapping a non-quantized model. Not used here.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
from huggingface_hub import snapshot_download

snapshot_download("Manoj2142/taxops-dev", repo_type="dataset", local_dir="/kaggle/temp/taxops",
                  allow_patterns=["taxops/**", "scripts/**"])
os.chdir("/kaggle/temp/taxops")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
sys.exit(subprocess.run(RUN, shell=True).returncode)
