import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

COMBINED = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/sft_diff_combined.jsonl")
HF_REPO  = "timchen0618/browsecomp-plus-sft-diff-v1"

api = HfApi()
# Delete and recreate the repo for a clean slate
print("Deleting dataset repo...")
api.delete_repo(repo_id=HF_REPO, repo_type="dataset", missing_ok=True)
api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=False)
print("Repo recreated.")

rows = [json.loads(l) for l in COMBINED.open() if l.strip()]
print(f"Loaded {len(rows)} rows, columns: {list(rows[0].keys())}")
ds = Dataset.from_list(rows)
print(f"Dataset: {ds}")
ds.push_to_hub(HF_REPO, split="train")
print("Done.")
