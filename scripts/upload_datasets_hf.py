#!/usr/bin/env python3
"""Upload RAPT dataset directories to a HuggingFace dataset repo.

Requires a one-time login: ``huggingface-cli login`` (or ``hf auth login``).

Example:
  python scripts/upload_datasets_hf.py datasets/g1_velocity datasets/g1_mimic_* \\
      --repo hmunn/rapt-g1-ood --card datasets/DATASET_CARD.md --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="+", help="dataset directories to upload")
    ap.add_argument("--repo", required=True, help="dataset repo id, e.g. user/rapt-g1-ood")
    ap.add_argument("--card", help="markdown file to upload as the dataset card (README.md)")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Logged in as {user}")
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    for ds in args.datasets:
        ds = Path(ds)
        if not (ds / "train.npz").exists():
            print(f"skip {ds}: no train.npz")
            continue
        print(f"Uploading {ds} -> {args.repo}/{ds.name} ...")
        api.upload_folder(
            folder_path=str(ds),
            path_in_repo=ds.name,
            repo_id=args.repo,
            repo_type="dataset",
            allow_patterns=["*.npz", "metadata.json"],
        )

    if args.card:
        api.upload_file(
            path_or_fileobj=args.card,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
