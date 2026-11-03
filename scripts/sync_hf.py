"""Upload the project code to the private dev repo on Hugging Face (Colab/HF jobs pull from it)."""

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
IGNORE = ["vendor/**", ".venv/**", "out/**", "runs/**", ".git/**", "**/__pycache__/**", "*.pyc",
          "**/.DS_Store", ".DS_Store", ".pytest_cache/**"]

if __name__ == "__main__":
    api = HfApi()
    api.upload_folder(repo_id="Manoj2142/taxops-dev", repo_type="dataset", folder_path=str(ROOT),
                      ignore_patterns=IGNORE, commit_message="sync code")
    for f in [f for f in api.list_repo_files("Manoj2142/taxops-dev", repo_type="dataset") if f.startswith(".pytest_cache")]:
        api.delete_file(f, "Manoj2142/taxops-dev", repo_type="dataset")
    print("synced to https://huggingface.co/datasets/Manoj2142/taxops-dev (private)")
