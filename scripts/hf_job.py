"""Run a project command on a Hugging Face Jobs GPU (paid from the account's HF credit).

The job pulls the code from the private dev repo, runs the command, and uploads
runs/<name> back to the repo.

    python scripts/hf_job.py --name base "python scripts/eval_model.py --limit 30 --out runs/base"
    python scripts/hf_job.py --flavor l4x1 --timeout 1h --name sft-smoke "python scripts/train_sft.py --smoke --out runs/sft-smoke"

Prints the job URL; follow with:  hf jobs logs <job_id>
"""

import argparse
import shlex

from huggingface_hub import HfApi, get_token

REPO = "Manoj2142/taxops-dev"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--name", required=True, help="results go to runs/<name> in the repo")
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="1h")
    ap.add_argument("--pip", default="transformers peft accelerate datasets huggingface_hub")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--agenttune", action="store_true",
                    help="Python 3.12 image + AgentTune from GitHub (needs torch 2.11), for GRPO")
    ap.add_argument("--vllm", action="store_true", help="with --agenttune: install the [vllm] extra")
    args = ap.parse_args()

    if args.agenttune:
        args.image = "ghcr.io/astral-sh/uv:python3.12-bookworm"
        extra = "[vllm]" if args.vllm else ""
        setup = ("apt-get update -qq && apt-get install -y -qq git >/dev/null && "
                 f"uv pip install --system -q 'agenttune{extra} @ git+https://github.com/Lexsi-Labs/AgentTune' huggingface_hub")
    else:
        setup = f"pip install -q -U {args.pip}"
    script = f"""set -e
{setup}
python -c "from huggingface_hub import snapshot_download; snapshot_download('{REPO}', repo_type='dataset', local_dir='/work')"
cd /work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/{args.name}
set +e
{args.command} 2>&1 | tee runs/{args.name}/job.log
status=${{PIPESTATUS[0]}}
set -e
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='{REPO}', repo_type='dataset', folder_path='runs/{args.name}', path_in_repo='runs/{args.name}', commit_message='job {args.name}')"
exit $status
"""
    job = HfApi().run_job(image=args.image, command=["bash", "-c", script], flavor=args.flavor,
                          secrets={"HF_TOKEN": get_token()}, timeout=args.timeout)
    print("job id:", job.id)
    print("url:", getattr(job, "url", None))
    print("flavor:", args.flavor, "| timeout:", args.timeout, "| command:", shlex.quote(args.command))


if __name__ == "__main__":
    main()
