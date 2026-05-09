"""
Rollout daemon — runs on GPU 3.

Each iteration:
  1. Check for ckpt_ready.flag → reload LoRA weights into vLLM if present.
  2. Sample a batch of queries.
  3. Generate G=cfg["group_size"] trajectories per query (asyncio, concurrent).
  4. Stream each completed trajectory to GPT-OSS-120B on GPU 2 for final answer.
  5. Kill explorer vLLM server, start Qwen3-32B judge server on GPU 3.
  6. Batch-score all answers with the judge.
  7. Kill judge server, restart explorer server.
  8. Push (trajectory, reward) pairs to the filesystem buffer.

Usage:
  python rl/trl/rollout_daemon.py --config rl/trl/config.yaml --gpu-id 3
"""

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


import aiohttp
import yaml

# Project root on sys.path so we can import searcher/
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from rl.trl.buffer import append_sample, check_and_clear_flag
from rl.trl.reward import call_main_agent, call_judge
from rl.trl.rollout_worker import generate_trajectory


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queries(tsv_path: str) -> list[tuple[str, str]]:
    queries = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))
    return queries


def load_ground_truth(jsonl_path: str) -> dict[str, str]:
    gt = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["query_id"])] = obj["answer"]
    return gt


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(
    model_path: str,
    port: int,
    gpu_id: int,
    gpu_memory_utilization: float = 0.90,
    extra_args: list[str] | None = None,
    lora_modules: str | None = None,
) -> subprocess.Popen:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if lora_modules:
        cmd += ["--enable-lora", "--lora-modules", lora_modules]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.Popen(cmd, env=env)


def wait_for_server(port: int, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"vLLM server on port {port} did not start within {timeout:.0f}s")


def kill_server(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# One rollout iteration
# ---------------------------------------------------------------------------

async def _rollout_iteration(
    queries: list[tuple[str, str]],
    ground_truth: dict[str, str],
    cfg: dict,
    searcher,
    iteration: int,
) -> list[dict]:
    G = cfg["group_size"]
    batch_queries = cfg["batch_queries"]

    # Cycle through queries round-robin
    start = (iteration * batch_queries) % len(queries)
    batch = queries[start : start + batch_queries]
    if len(batch) < batch_queries:
        batch += queries[: batch_queries - len(batch)]

    # --- Step 1: explorer trajectories (all concurrent) ---
    print(
        f"[daemon] iter={iteration} generating {len(batch) * G} trajectories ...",
        flush=True,
    )
    async with aiohttp.ClientSession() as session:
        groups = await asyncio.gather(*[
            asyncio.gather(*[
                generate_trajectory(query, qid, searcher, cfg, session)
                for _ in range(G)
            ])
            for qid, query in batch
        ])

    # --- Step 2: main agent completions (GPU 2, stream as each group completes) ---
    print(f"[daemon] iter={iteration} calling main agent ...", flush=True)
    flat_trajs = [t for group in groups for t in group]
    async with aiohttp.ClientSession() as session:
        answers = await asyncio.gather(*[
            call_main_agent(
                t["messages"], t["query_id"], t["query"], cfg, session, searcher
            )
            for t in flat_trajs
        ])

    # --- Step 3: judge all answers (GPU 3 is now in judge mode) ---
    print(f"[daemon] iter={iteration} scoring {len(answers)} answers ...", flush=True)
    async with aiohttp.ClientSession() as session:
        rewards = await asyncio.gather(*[
            call_judge(
                t["query"],
                ans,
                ground_truth.get(str(t["query_id"]), ""),
                cfg,
                session,
            )
            for t, ans in zip(flat_trajs, answers)
        ])

    # --- Step 4: pack buffer samples ---
    samples = []
    for traj, answer, reward in zip(flat_trajs, answers, rewards):
        samples.append({
            "query_id": traj["query_id"],
            "query": traj["query"],
            "messages": traj["messages"],
            "final_answer": answer,
            "reward": reward,
            "group_size": G,
        })

    n_correct = sum(rewards)
    print(
        f"[daemon] iter={iteration} done: {n_correct}/{len(rewards)} correct",
        flush=True,
    )
    return samples


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Stop after N rollout iterations (default: run forever).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root
    for key in ("index_path", "train_queries", "ground_truth", "buffer_dir", "ckpt_flag"):
        cfg[key] = str(_ROOT / cfg[key])

    # Initialise FAISS searcher
    from searcher.searchers.faiss_searcher import FaissSearcher

    class _SearcherArgs:
        index_path = cfg["index_path"]
        model_name = cfg.get("index_model_name", "Qwen/Qwen3-Embedding-8B")
        normalize = True
        pooling = "eos"
        torch_dtype = "float16"
        dataset_name = "Tevatron/browsecomp-plus-corpus"
        task_prefix = (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:"
        )
        max_length = 8192
        attn_implementation = "flash_attention_2"

    print("[daemon] loading FAISS index + embedding model (may take several minutes) ...", flush=True)
    searcher = FaissSearcher(_SearcherArgs())
    print("[daemon] searcher ready", flush=True)

    queries = load_queries(cfg["train_queries"])
    ground_truth = load_ground_truth(cfg["ground_truth"])
    buffer_dir = Path(cfg["buffer_dir"])
    ckpt_flag = Path(cfg["ckpt_flag"])
    ckpt_dir = Path(cfg["ckpt_dir"])
    gpu_id = args.gpu_id

    print(
        f"[daemon] {len(queries)} queries | {len(ground_truth)} ground-truth entries",
        flush=True,
    )

    # --- Start explorer vLLM server on GPU 3 ---
    explorer_model = cfg["sft_checkpoint"]  # LoRA weights loaded via lora_modules
    lora_modules = f"explorer={explorer_model}"
    # TODO: once the checkpoint path is real, load LoRA via lora_modules arg.
    # For now, load the base model directly.
    explorer_proc = start_vllm_server(
        cfg["explorer_model"],
        cfg["rollout_port"],
        gpu_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"[daemon] starting explorer vLLM (port {cfg['rollout_port']}) ...", flush=True)
    wait_for_server(cfg["rollout_port"])
    print("[daemon] explorer server ready", flush=True)

    judge_proc: subprocess.Popen | None = None
    iteration = 0

    try:
        while True:
            # Check for new learner checkpoint
            if check_and_clear_flag(ckpt_flag):
                print(f"[daemon] new checkpoint at {ckpt_dir}, reloading ...", flush=True)
                # Kill current explorer server and restart with updated weights
                kill_server(explorer_proc)
                # TODO: pass updated LoRA via --lora-modules once weight format is set
                explorer_proc = start_vllm_server(
                    cfg["explorer_model"], cfg["rollout_port"], gpu_id,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                wait_for_server(cfg["rollout_port"])
                print("[daemon] explorer reloaded", flush=True)

            # Run trajectory generation + main agent phase (explorer on GPU 3)
            # We generate trajectories first, then do main-agent calls while
            # still on explorer mode (main agent lives on GPU 2, not GPU 3).
            # Only after all main-agent answers are collected do we swap GPU 3.

            # Generate trajectories and get main-agent answers (explorer still on GPU 3)
            G = cfg["group_size"]
            batch_queries_n = cfg["batch_queries"]
            start = (iteration * batch_queries_n) % len(queries)
            batch = queries[start : start + batch_queries_n]
            if len(batch) < batch_queries_n:
                batch += queries[: batch_queries_n - len(batch)]

            print(
                f"[daemon] iter={iteration}: generating {len(batch) * G} trajectories",
                flush=True,
            )
            async def _gen_and_main():
                async with aiohttp.ClientSession() as session:
                    groups = await asyncio.gather(*[
                        asyncio.gather(*[
                            generate_trajectory(query, qid, searcher, cfg, session)
                            for _ in range(G)
                        ])
                        for qid, query in batch
                    ])
                flat = [t for g in groups for t in g]
                print(f"[daemon] iter={iteration}: calling main agent ...", flush=True)
                async with aiohttp.ClientSession() as session:
                    answers = await asyncio.gather(*[
                        call_main_agent(
                            t["messages"], t["query_id"], t["query"],
                            cfg, session, searcher,
                        )
                        for t in flat
                    ])
                return groups, flat, answers

            groups, flat_trajs, answers = asyncio.run(_gen_and_main())

            # Swap GPU to judge
            print(f"[daemon] swapping GPU {gpu_id} to judge mode ...", flush=True)
            kill_server(explorer_proc)
            judge_proc = start_vllm_server(
                cfg["judge_model"],
                cfg["rollout_port"],
                gpu_id,
                gpu_memory_utilization=args.gpu_memory_utilization,
                extra_args=["--quantization", "int8"],
            )
            wait_for_server(cfg["rollout_port"])
            print("[daemon] judge server ready", flush=True)

            # Score answers
            print(f"[daemon] iter={iteration}: scoring {len(flat_trajs)} answers ...", flush=True)
            async def _score():
                async with aiohttp.ClientSession() as session:
                    return await asyncio.gather(*[
                        call_judge(
                            t["query"],
                            ans,
                            ground_truth.get(str(t["query_id"]), ""),
                            cfg,
                            session,
                        )
                        for t, ans in zip(flat_trajs, answers)
                    ])

            rewards = asyncio.run(_score())

            # Swap GPU back to explorer
            print(f"[daemon] swapping GPU {gpu_id} back to explorer ...", flush=True)
            kill_server(judge_proc)
            judge_proc = None
            explorer_proc = start_vllm_server(
                cfg["explorer_model"], cfg["rollout_port"], gpu_id,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            wait_for_server(cfg["rollout_port"])

            # Push to buffer
            for traj, answer, reward in zip(flat_trajs, answers, rewards):
                append_sample(buffer_dir, {
                    "query_id": traj["query_id"],
                    "query": traj["query"],
                    "messages": traj["messages"],
                    "final_answer": answer,
                    "reward": reward,
                    "group_size": G,
                })

            n_correct = sum(rewards)
            print(
                f"[daemon] iter={iteration}: pushed {len(flat_trajs)} samples "
                f"({n_correct}/{len(flat_trajs)} correct)",
                flush=True,
            )
            iteration += 1
            if args.max_iterations is not None and iteration >= args.max_iterations:
                print(
                    f"[daemon] max-iterations={args.max_iterations} reached, exiting.",
                    flush=True,
                )
                break

    finally:
        kill_server(explorer_proc)
        if judge_proc:
            kill_server(judge_proc)


if __name__ == "__main__":
    main()
