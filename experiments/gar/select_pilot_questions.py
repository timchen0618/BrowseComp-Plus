"""Select 100 stratified pilot questions for GAR experiment.

Samples 50 correct + 50 incorrect from existing WebExplorer baselines
(string matching). Outputs a fixed set reused across all conditions.
"""

import json
import os
import random
import argparse
from pathlib import Path


def extract_answer_from_trajectory(traj: dict) -> str:
    """Extract the agent's final answer from trajectory result array."""
    for entry in reversed(traj.get("result", [])):
        if entry.get("type") == "output_text":
            return entry.get("output", "").strip()
    return ""


def string_match(prediction: str, gold: str) -> bool:
    """Case-insensitive containment check."""
    return gold.lower().strip() in prediction.lower().strip()


def main():
    parser = argparse.ArgumentParser(description="Select pilot questions for GAR experiment")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default="/scratch/afw8937/efficient-search-agents/runs/qwen3-8/webexplorer",
        help="Directory containing WebExplorer trajectory JSON files",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="/scratch/afw8937/efficient-search-agents/data/browsecomp_plus_decrypted.jsonl",
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: experiments/gar/pilot_100.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-correct", type=int, default=50, help="Number of correct questions to sample")
    parser.add_argument("--n-incorrect", type=int, default=50, help="Number of incorrect questions to sample")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent / "pilot_100.jsonl")

    random.seed(args.seed)

    # Load ground truth
    gt = {}
    with open(args.ground_truth) as f:
        for line in f:
            entry = json.loads(line)
            gt[str(entry["query_id"])] = {
                "query": entry["query"],
                "answer": entry["answer"],
            }
    print(f"Loaded {len(gt)} ground truth entries")

    # Load trajectories and classify
    correct_ids = []
    incorrect_ids = []
    traj_dir = Path(args.trajectory_dir)
    for json_path in sorted(traj_dir.glob("run_*.json")):
        with open(json_path) as f:
            traj = json.load(f)
        qid = str(traj.get("query_id", ""))
        if qid not in gt:
            continue
        if traj.get("status") != "completed":
            incorrect_ids.append(qid)
            continue
        prediction = extract_answer_from_trajectory(traj)
        gold = gt[qid]["answer"]
        if string_match(prediction, gold):
            correct_ids.append(qid)
        else:
            incorrect_ids.append(qid)

    print(f"Correct: {len(correct_ids)}, Incorrect: {len(incorrect_ids)}")

    # Stratified sampling
    if len(correct_ids) < args.n_correct:
        print(f"Warning: only {len(correct_ids)} correct, sampling all")
        sampled_correct = correct_ids
    else:
        sampled_correct = random.sample(correct_ids, args.n_correct)

    if len(incorrect_ids) < args.n_incorrect:
        print(f"Warning: only {len(incorrect_ids)} incorrect, sampling all")
        sampled_incorrect = incorrect_ids
    else:
        sampled_incorrect = random.sample(incorrect_ids, args.n_incorrect)

    # Build output
    pilot = []
    for qid in sampled_correct:
        pilot.append({
            "query_id": qid,
            "query": gt[qid]["query"],
            "answer": gt[qid]["answer"],
            "baseline_correct": True,
        })
    for qid in sampled_incorrect:
        pilot.append({
            "query_id": qid,
            "query": gt[qid]["query"],
            "answer": gt[qid]["answer"],
            "baseline_correct": False,
        })

    # Shuffle to avoid ordering bias
    random.shuffle(pilot)

    # Also write a TSV for tongyi_client.py compatibility
    tsv_path = str(args.output).replace(".jsonl", ".tsv")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for entry in pilot:
            f.write(json.dumps(entry) + "\n")
    with open(tsv_path, "w") as f:
        for entry in pilot:
            f.write(f"{entry['query_id']}\t{entry['query']}\n")

    print(f"Wrote {len(pilot)} pilot questions to {args.output}")
    print(f"Wrote TSV to {tsv_path}")
    print(f"  Correct: {len(sampled_correct)}, Incorrect: {len(sampled_incorrect)}")


if __name__ == "__main__":
    main()
