# GRPO Training — TRL Implementation

Train the SFT-ed Qwen3.5-4B explorer model with GRPO on 4× H200 GPUs using a
single SBATCH job.

---

## File Layout

```
rl/trl/
├── config.yaml             ← all hyperparams; pass to every script
├── buffer.py               ← filesystem JSONL buffer (actor ↔ learner)
├── rollout_worker.py       ← multi-turn Qwen3.5-4B trajectory generator
├── reward.py               ← main-agent + judge async HTTP calls
├── rollout_daemon.py       ← actor loop: explorer → main agent → judge → buffer
├── grpo_train.py           ← GRPO learner (FSDP, custom training loop)
├── grpo_train.SBATCH       ← 4× H200 SLURM job
└── tests/
    ├── test_buffer.py          ← buffer unit tests (17 tests)
    ├── test_reward.py          ← reward parsing unit tests (18 tests)
    ├── test_rollout_worker.py  ← rollout worker unit tests (19 tests)
    └── test_smoke.py           ← import + config + GRPO math tests (22 tests)
```

---

## GPU Layout

| GPU | Role | Model | Notes |
|-----|------|-------|-------|
| 0, 1 | Learner | Qwen3.5-4B LoRA, FSDP | GRPO gradient steps |
| 2 | Main agent | GPT-OSS-120B vLLM TP=1 | Always loaded; `gpu_memory_utilization=0.85` |
| 3 | Actor (time-muxed) | Qwen3.5-4B → Qwen3-32B | Explorer rollout then judge; alternating |

---

## Quick Start

### 1. Update `config.yaml`

Set `sft_checkpoint` to the actual SFT LoRA checkpoint path once available.

### 2. Submit the job

```bash
sbatch rl/trl/grpo_train.SBATCH
```

The SBATCH script starts three processes in parallel:
1. **GPU 2** — `vllm serve openai/gpt-oss-120b` (persistent throughout job)
2. **GPU 3** — `rollout_daemon.py` (manages explorer↔judge model swap)
3. **GPU 0,1** — `accelerate launch grpo_train.py` (blocks until buffer fills)

### 3. Monitor

```bash
tail -f sbatch_outputs/grpo_explorer.out
```

Checkpoints are saved to `rl/trl/tmp/rl_buffer/ckpt_latest/` every
`checkpoint_every=20` steps. The final model lands in `rl/trl/runs/grpo_run1/`.

---

## Config Reference

| Key | Default | Description |
|-----|---------|-------------|
| `sft_checkpoint` | `PLACEHOLDER` | Path to SFT LoRA checkpoint |
| `index_path` | `indexes/Qwen3-Embedding-8B/*` | FAISS index glob |
| `train_queries` | `topics-qrels/bcp/queries_train680.tsv` | 680 training queries |
| `ground_truth` | `data/browsecomp_plus_decrypted_train680.jsonl` | Ground-truth answers |
| `main_agent_port` | `8010` | GPU 2 vLLM port |
| `rollout_port` | `8011` | GPU 3 vLLM port (shared by explorer + judge) |
| `explorer_model` | `Qwen/Qwen3.5-4B` | Explorer base model |
| `judge_model` | `Qwen/Qwen3-32B` | Judge model (int8 on GPU 3) |
| `group_size` | `4` | G — trajectories per query |
| `batch_queries` | `32` | Queries per rollout iteration (→ 128 samples) |
| `kl_beta` | `0.01` | KL penalty coefficient |
| `learning_rate` | `1e-5` | AdamW learning rate |
| `max_steps` | `500` | Total training steps |
| `checkpoint_every` | `20` | Steps between weight-sync checkpoints |
| `min_buffer_size` | `64` | Learner blocks until buffer reaches this |
| `max_turns` | `15` | Max search turns per explorer trajectory |
| `buffer_dir` | `rl/trl/tmp/rl_buffer` | Filesystem buffer directory |

---

## Running Tests

```bash
# From project root inside Singularity:
python -m pytest rl/trl/tests/ -v
```

All 76 tests run without a GPU or any live services.

```
test_buffer.py          17 tests   buffer I/O, atomicity, flag protocol
test_reward.py          18 tests   judge response parsing, answer extraction
test_rollout_worker.py  19 tests   tool call execution, answer extraction, trajectory gen (mocked vLLM)
test_smoke.py           22 tests   imports, config completeness, GRPO math, collate
```

---

## Architecture

### Pipeline Flow

```
Query TSV
    │
    ▼  GPU 3, Qwen3.5-4B vLLM (rollout_worker.py)
[Explorer rollout — G=4 trajectories per query, concurrent asyncio]
    │
    ▼  GPU 2, GPT-OSS-120B vLLM (reward.py → call_main_agent)
[Main agent — traj_orig_ext mode, search tools available]
    │
    ▼  GPU 3 swaps to Qwen3-32B int8 (reward.py → call_judge)
[Judge — binary {0,1} reward per trajectory]
    │
    ▼  buffer.py (one file per sample, atomic rename)
[Filesystem buffer — rl/trl/tmp/rl_buffer/sample_*.jsonl]
    │
    ▼  GPU 0,1 FSDP (grpo_train.py)
[Learner — GRPO loss + KL, optimizer step, checkpoint every K steps]
```

### GRPO Loss

```
A_i = (r_i − mean(r_group)) / (std(r_group) + ε)    # per-group normalisation

L = −mean_i(A_i · Σ_t log p_θ(a_t | a_{<t}) · mask_t / |mask|)
    + β · mean_i((Σ_t log p_θ · mask_t − Σ_t log p_ref · mask_t) / |mask|)
```

Where `mask` is 1 only for assistant-turn tokens (same as SFT loss mask).

### Buffer Protocol

- **Write** (`rollout_daemon.py`): `append_sample()` — writes one file per sample via atomic `tmp → rename`, so the learner never reads a partial file.
- **Read** (`grpo_train.py`): `pop_samples(n)` — blocks until `n` files exist, reads and deletes them.
- **Weight sync**: learner saves LoRA weights to `ckpt_latest/`, then touches `ckpt_ready.flag`. Daemon checks flag at each iteration boundary, reloads weights, deletes flag.

---

## End-to-End Testing Scheme

These tests require real hardware but validate the full pipeline.

### E2E-1: Buffer integration test (no GPU required)

```bash
# Terminal 1 — writer (simulates rollout daemon)
python - << 'EOF'
from rl.trl.buffer import append_sample
import time, json
for i in range(50):
    append_sample("rl/trl/tmp/test_e2e_buf",
                  {"query_id": f"Q{i}", "reward": i % 2, "messages": []})
    time.sleep(0.05)
print("Writer done")
EOF

# Terminal 2 — reader (simulates learner)
python - << 'EOF'
from rl.trl.buffer import pop_samples
samples = pop_samples("rl/trl/tmp/test_e2e_buf", 50, timeout=60)
print(f"Got {len(samples)} samples, rewards: {[s['reward'] for s in samples]}")
EOF
```

Expected: reader receives all 50 samples in insertion order, rewards alternate 0/1.

### E2E-2: Rollout worker against a live vLLM server

1. Start a small model (e.g. `Qwen/Qwen3.5-0.5B`) on port 8011:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3.5-0.5B --port 8011 --gpu-memory-utilization 0.8
   ```
2. Create a minimal FAISS index or mock searcher:
   ```python
   # mock_searcher.py
   class MockSearcher:
       def search(self, q, k): return [{"docid": "d1", "score": 0.9, "text": "test"}]
   ```
3. Run one trajectory:
   ```python
   import asyncio, yaml
   from rl.trl.rollout_worker import generate_trajectory
   import aiohttp

   cfg = yaml.safe_load(open("rl/trl/config.yaml"))
   cfg["explorer_model"] = "Qwen/Qwen3.5-0.5B"  # override to tiny model
   searcher = MockSearcher()

   async def run():
       async with aiohttp.ClientSession() as sess:
           result = await generate_trajectory("What is 2+2?", "Q0", searcher, cfg, sess)
       print("Terminated:", result["terminated"])
       print("Turns:", len([m for m in result["messages"] if m["role"] == "assistant"]))

   asyncio.run(run())
   ```

Expected: trajectory completes (terminated=True), messages list contains alternating assistant/user turns.

### E2E-3: Reward pipeline (mock judge)

1. Start a small model on port 8011 (same as E2E-2 but in "judge" role).
2. Run `call_judge` directly:
   ```python
   import asyncio, yaml, aiohttp
   from rl.trl.reward import call_judge

   cfg = yaml.safe_load(open("rl/trl/config.yaml"))
   cfg["judge_model"] = "Qwen/Qwen3.5-0.5B"

   async def run():
       async with aiohttp.ClientSession() as sess:
           reward = await call_judge(
               "What is the capital of France?", "Paris", "Paris", cfg, sess
           )
       print("Reward:", reward)  # expect 1

   asyncio.run(run())
   ```

### E2E-4: One full training step (single GPU, toy model)

Verify the learner can do one step end-to-end on a single GPU with a tiny model:

```python
# Manually push 128 fake samples, then run grpo_train.py for 1 step
import yaml
from rl.trl.buffer import append_sample

cfg = yaml.safe_load(open("rl/trl/config.yaml"))
cfg["max_steps"] = 1
cfg["batch_queries"] = 4
cfg["group_size"] = 4
cfg["min_buffer_size"] = 16

# Push 16 fake samples
for i in range(16):
    append_sample(cfg["buffer_dir"], {
        "query_id": f"Q{i // 4}",
        "reward": float(i % 2),
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<answer>4</answer>"},
        ],
        "final_answer": "4",
        "group_size": 4,
    })
```

Then launch the learner:
```bash
CUDA_VISIBLE_DEVICES=0 python rl/trl/grpo_train.py --config rl/trl/config.yaml
```

Expected: one training step completes, loss printed, no OOM.

### E2E-5: Full 4-GPU job (production)

```bash
sbatch rl/trl/grpo_train.SBATCH
# Monitor
tail -f sbatch_outputs/grpo_explorer.out
```

Check:
- [ ] Main agent server starts within 5 minutes
- [ ] Rollout daemon starts generating trajectories
- [ ] Buffer fills to `min_buffer_size` (64) before learner starts
- [ ] Learner logs first training step
- [ ] Checkpoint written at step 20
- [ ] Daemon detects `ckpt_ready.flag` and logs "reloading"
- [ ] Job completes at `max_steps=500`; final model in `rl/trl/runs/grpo_run1/`

---

## Known TODOs

- `grpo_train.py`: Load actual SFT checkpoint weights once `cfg["sft_checkpoint"]` is set.
- `rollout_daemon.py`: Pass updated LoRA via `--lora-modules` when reloading after checkpoint.
- `grpo_train.py`: The warm-up `pop_samples` call at startup drains samples before the first training batch; adjust to `peek_samples` + separate drain for cleaner logic.
- `rollout_daemon.py`: The `_gen_and_main` nested async function is defined inside a sync loop; refactor to a top-level async function.
- Consider increasing `group_size` to 8 once memory budget is confirmed.
