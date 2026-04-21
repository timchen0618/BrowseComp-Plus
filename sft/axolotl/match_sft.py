"""
Match original excerpt records with their converted messages counterparts
for both gpt-oss and qwen templates. Outputs one combined JSONL with
query_id, split, excerpt, messages_json (gpt-oss), messages_json_qwen.
"""
import json
from pathlib import Path

TRAJ_FOLDER = Path("/scratch/hc3337/projects/BrowseComp-Plus/runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4")
ORIGINAL    = Path("/scratch/hc3337/projects/BrowseComp-Plus/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl")
TRAIN_GPT   = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/train.jsonl")
VAL_GPT     = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/val.jsonl")
TRAIN_QWEN  = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data_qwen/train.jsonl")
VAL_QWEN    = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data_qwen/val.jsonl")
OUTPUT      = Path("/scratch/hc3337/projects/BrowseComp-Plus/sft/axolotl/data/sft_diff_combined.jsonl")

def build_index(paths):
    idx = {}  # first_user_content -> messages list
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get("messages", [])
                if msgs and msgs[0].get("role") == "user":
                    idx[msgs[0]["content"]] = msgs
    return idx

gpt_idx  = build_index([TRAIN_GPT,  VAL_GPT])
qwen_idx = build_index([TRAIN_QWEN, VAL_QWEN])
print(f"gpt-oss index: {len(gpt_idx)}  qwen index: {len(qwen_idx)}")

traj_cache = {}
matched = no_traj = no_match_either = 0

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
            no_traj += 1
            continue
        om = traj.get("original_messages", [])
        if not om:
            no_traj += 1
            continue
        key = om[0].get("content", "")
        msgs_gpt  = gpt_idx.get(key)
        msgs_qwen = qwen_idx.get(key)
        if msgs_gpt is None and msgs_qwen is None:
            no_match_either += 1
            continue
        out.write(json.dumps({
            "query_id":          str(rec.get("query_id", "")),
            "excerpt":           rec.get("excerpt", ""),
            "messages_json":     json.dumps(msgs_gpt,  ensure_ascii=False) if msgs_gpt  else None,
            "messages_json_qwen":json.dumps(msgs_qwen, ensure_ascii=False) if msgs_qwen else None,
        }, ensure_ascii=False) + "\n")
        matched += 1

print(f"matched={matched}  no_traj={no_traj}  no_match_either={no_match_either}")
print(f"gpt-only={(sum(1 for _ in open(OUTPUT)) if OUTPUT.exists() else '?')} rows written")
