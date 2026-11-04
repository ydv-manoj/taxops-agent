# TaxOps eval on a Kaggle GPU. Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# Untrained Tiny Aya ("before" numbers): 60 eval cases + the same 30 shift cases Claude ran.
RUN = ("python scripts/eval_model.py --split eval --limit 60 --out /kaggle/working/runs/base-60 && "
       "python scripts/eval_model.py --split eval-shift --limit 30 --out /kaggle/working/runs/base-shift-30")


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
