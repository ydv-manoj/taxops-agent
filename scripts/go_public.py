"""Make the TaxOps submission public on Hugging Face (the GitHub repo already is).

    python scripts/go_public.py          # show what would change
    python scripts/go_public.py --yes    # do it

taxops-dev (the private code mirror Kaggle reads) and the v1/v2/v4 adapters stay private.
"""

import sys

from huggingface_hub import HfApi, get_token

REPOS = [("Manoj2142/taxops-tinyaya-sft-v6", "model"), ("Manoj2142/taxops-tinyaya-sft-v3", "model"),
         ("Manoj2142/taxops-traces", "dataset"), ("Manoj2142/taxops-replay", "space")]
COLLECTIONS = ["Manoj2142/taxops-models-6af09c4cebb37eb4c911abb5",
               "Manoj2142/taxops-data-and-traces-6af09c4e19339ae58deca1c1"]


def main() -> None:
    go = "--yes" in sys.argv
    api = HfApi(token=get_token())
    for repo, kind in REPOS:
        print(f"{kind:8} {repo}: private={api.repo_info(repo, repo_type=kind).private}")
        if go:
            api.update_repo_settings(repo, repo_type=kind, private=False)
    for slug in COLLECTIONS:
        print(f"collection {slug}: private={api.get_collection(slug).private}")
        if go:
            api.update_collection_metadata(slug, private=False)
    print("now public" if go else "dry run; rerun with --yes")


if __name__ == "__main__":
    main()
