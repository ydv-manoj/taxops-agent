# TaxOps SFT training on a Kaggle T4 (QLoRA). Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# The token is read-only, so the adapter is not pushed: it lands in /kaggle/working and is
# downloaded with `kaggle kernels output`, then uploaded to HF from the laptop.
SMOKE = True
RUN = ("python scripts/train_sft.py --qlora --smoke --out /kaggle/working/sft-v4-smoke" if SMOKE else
       "python scripts/train_sft.py --qlora --out /kaggle/working/sft-v4 && "
       "python scripts/eval_model.py --adapter /kaggle/working/sft-v4 --split eval --limit 30 "
       "--out /kaggle/working/runs/sft-v4-eval-30")


# Read-only HF token from the private dataset papw123/taxops-hf-token (CLI-pushed runs
# can't read Kaggle Secrets). Falls back to the secret for runs started in the editor.
found = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
if found:
    os.environ["HF_TOKEN"] = open(found[0]).read().strip()
else:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft", "accelerate",
                "huggingface_hub", "bitsandbytes"], check=True)
from huggingface_hub import snapshot_download

snapshot_download("Manoj2142/taxops-dev", repo_type="dataset", local_dir="/kaggle/temp/taxops",
                  allow_patterns=["taxops/**", "scripts/**"])
os.chdir("/kaggle/temp/taxops")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
sys.exit(subprocess.run(RUN, shell=True).returncode)
