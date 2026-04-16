"""Upload sft_diff_combined.jsonl to HuggingFace as a dataset."""
import json
from pathlib import Path
from datasets import Dataset

COMBINED = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/sft_diff_combined.jsonl")
HF_REPO = "timchen0618/browsecomp-plus-sft-diff-v1"

rows = []
with COMBINED.open() as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Loaded {len(rows)} rows")
ds = Dataset.from_list(rows)
print(f"Dataset: {ds}")
ds.push_to_hub(HF_REPO, split="train")
print(f"Uploaded to {HF_REPO}")
