# TaxOps eval on a Kaggle GPU. Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# Earlier run, untrained Tiny Aya ("before" numbers):
#   eval_model.py --split eval --limit 60 ... && eval_model.py --split eval-shift --limit 30 ...
# Now SFT v4, from the papw123/taxops-train output (mounted under /kaggle/input), with eval on
# GPU 0 and eval-shift on GPU 1 at the same time.
ADAPTER = "$(dirname $(ls /kaggle/input/**/sft-v4/adapter_config.json /kaggle/input/*/*/*/sft-v4/adapter_config.json 2>/dev/null | head -1))"
RUN = (f"A={ADAPTER}; echo adapter=$A; "
       "CUDA_VISIBLE_DEVICES=0 python scripts/eval_model.py --adapter $A --split eval --out /kaggle/working/runs/sft-v4-eval-full > /kaggle/working/eval.log 2>&1 & P=$!; "
       "CUDA_VISIBLE_DEVICES=1 python scripts/eval_model.py --adapter $A --split eval-shift --out /kaggle/working/runs/sft-v4-shift > /kaggle/working/shift.log 2>&1; "
       "wait $P; tail -3 /kaggle/working/eval.log /kaggle/working/shift.log")


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
from huggingface_hub import snapshot_download

snapshot_download("Manoj2142/taxops-dev", repo_type="dataset", local_dir="/kaggle/temp/taxops",
                  allow_patterns=["taxops/**", "scripts/**"])
os.chdir("/kaggle/temp/taxops")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
sys.exit(subprocess.run(RUN, shell=True).returncode)
