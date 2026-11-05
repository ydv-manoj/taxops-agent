# TaxOps eval on a Kaggle GPU. Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# Earlier runs: untrained Tiny Aya (eval --limit 60, eval-shift --limit 30), then SFT v3/v4 on the full
# eval and eval-shift splits. Now SFT v6 on eval-shift, split across both T4s (every 2nd case each).
# `timeout` stops a shard before the weekly GPU quota runs out; traces are flushed after every case.
ADAPTER = "Manoj2142/taxops-tinyaya-sft-v6"
EVAL = f"timeout 220m python scripts/eval_model.py --adapter {ADAPTER} --split eval-shift"
RUN = (f"CUDA_VISIBLE_DEVICES=0 {EVAL} --shard 0/2 --out /kaggle/working/runs/sft-v6-shift-0 > /kaggle/working/shift-0.log 2>&1 & P=$!; "
       f"CUDA_VISIBLE_DEVICES=1 {EVAL} --shard 1/2 --out /kaggle/working/runs/sft-v6-shift-1 > /kaggle/working/shift-1.log 2>&1; "
       "wait $P; tail -n 12 /kaggle/working/shift-0.log /kaggle/working/shift-1.log")


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
