# TaxOps SFT training on Kaggle's 2xT4. Needs a read-only HF token (private dataset or Kaggle secret):
# Tiny Aya is gated, and the code + adapters live in a private HF repo until submission).
# The command to run is in RUN below; results land in /kaggle/working/runs/.
import glob, os, subprocess, sys

# The token is read-only, so the adapter is not pushed: it lands in /kaggle/working and is
# downloaded with `kaggle kernels output`, then uploaded to HF from the laptop.
# v6 (v5 was stopped for it): continue v3 (bf16 LoRA from an A10G) in fp16 on both T4s, with no 4-bit base, so the
# adapter is trained on the same weights it is evaluated on (v4's QLoRA wasn't). First a 2-step
# memory check on the longest conversations, then the run, then v3 and v6 on the dev-shift
# split, one GPU each, to choose between them without touching eval-shift.
V3 = "Manoj2142/taxops-tinyaya-sft-v3"
TRAIN = f"torchrun --nproc_per_node 2 scripts/train_sft.py --mix v6 --init-adapter {V3} --lr 5e-5"
EVAL = "python scripts/eval_model.py --split dev-shift"
RUN = (f"{TRAIN} --smoke --out /kaggle/working/sft-v6-smoke && "
       f"{TRAIN} --max-minutes 240 --out /kaggle/working/sft-v6 && "
       f"(CUDA_VISIBLE_DEVICES=0 {EVAL} --adapter {V3} --out /kaggle/working/runs/v3-dev-shift > /kaggle/working/v3-dev.log 2>&1 & P=$!; "
       f"CUDA_VISIBLE_DEVICES=1 {EVAL} --adapter /kaggle/working/sft-v6 --out /kaggle/working/runs/v6-dev-shift > /kaggle/working/v6-dev.log 2>&1; "
       "wait $P; tail -n 25 /kaggle/working/v3-dev.log /kaggle/working/v6-dev.log)")


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
# Kaggle ships torchao 0.10; new peft refuses it when wrapping a non-quantized model. Not used here.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
from huggingface_hub import snapshot_download

snapshot_download("Manoj2142/taxops-dev", repo_type="dataset", local_dir="/kaggle/temp/taxops",
                  allow_patterns=["taxops/**", "scripts/**"])
os.chdir("/kaggle/temp/taxops")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
sys.exit(subprocess.run(RUN, shell=True).returncode)
