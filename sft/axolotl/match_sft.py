"""
Match original excerpt records with their converted messages counterparts.
Matching key: messages[0]["content"] (user system+question) must equal
original_messages[0]["content"] from the source trajectory.
Outputs combined JSONL: query_id, split, excerpt, messages_json
"""
import json
from pathlib import Path

TRAJ_FOLDER = Path("/scratch/hc3337/projects/BrowseComp-Plus/runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4")
ORIGINAL = Path("/scratch/hc3337/projects/BrowseComp-Plus/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl")
TRAIN    = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/train.jsonl")
VAL      = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/val.jsonl")
OUTPUT   = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/sft_diff_combined.jsonl")

# ── Step 1: build index from converted files (key = first user message content)
converted = {}  # content_key -> {messages, split}
for split, path in [("train", TRAIN), ("val", VAL)]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            if msgs and msgs[0].get("role") == "user":
                key = msgs[0]["content"]
                converted[key] = {"messages": msgs, "split": split}

print(f"Converted index built: {len(converted)} entries")

# ── Step 2: walk original records, load source trajectory, match
matched = 0
missing_traj = 0
no_match = 0
traj_cache = {}

with ORIGINAL.open() as f, OUTPUT.open("w") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = rec.get("source_file", "")
        traj_path = TRAJ_FOLDER / src
        if traj_path not in traj_cache:
            try:
                traj_cache[traj_path] = json.loads(traj_path.read_text())
            except Exception:
                traj_cache[traj_path] = None
        traj = traj_cache[traj_path]
        if traj is None:
            missing_traj += 1
            continue
        om = traj.get("original_messages", [])
        if not om:
            missing_traj += 1
            continue
        key = om[0].get("content", "")
        if key not in converted:
            no_match += 1
            continue
        conv = converted[key]
        out.write(json.dumps({
            "query_id":     str(rec.get("query_id", "")),
            "split":        conv["split"],
            "excerpt":      rec.get("excerpt", ""),
            "messages_json": json.dumps(conv["messages"], ensure_ascii=False),
        }, ensure_ascii=False) + "\n")
        matched += 1

print(f"matched={matched}  missing_traj={missing_traj}  no_match={no_match}")
print(f"Output: {OUTPUT}")
