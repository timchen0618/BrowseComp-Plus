# 2-GPU Pipeline Validation

Validates the full pipeline (explorer → main agent → judge → buffer → learner)
using tiny 0.5B models on 2 GPUs. Completes in ~5 minutes.

**Config file:** `rl/trl/config_2gpu_val.yaml`

GPU layout for this validation:

| GPU | What runs |
|-----|-----------|
| 0 | Main agent vLLM (Terminal 1) + Learner (Terminal 3) — share GPU, fine at 0.5B |
| 1 | Rollout daemon — manages explorer vLLM + judge vLLM, time-muxed (Terminal 2) |

---

## Before you start

Open **3 terminal windows**, all from the project root inside Singularity:

```bash
singularity exec --nv \
    --overlay /scratch/hc3337/envs/bcp.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash
source /ext3/env.sh
cd /scratch/hc3337/projects/BrowseComp-Plus
```

Then clear any leftover buffer files from a previous run:

```bash
rm -rf rl/trl/tmp/val_2gpu_buffer
```

---

## Step 1 — Start the main agent (Terminal 1)

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-0.8B \
    --port 8010 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4
```

**Wait for:** `Application startup complete.` in the terminal output.

**What it does:** Starts the main agent server on GPU 0, port 8010. Leaves plenty
of GPU 0 memory free for the learner in Step 3.

**Do not proceed to Step 2 until this is ready.**

---

## Step 2 — Start the rollout daemon (Terminal 2)

```bash
python rl/trl/rollout_daemon.py \
    --config rl/trl/config_2gpu_val.yaml \
    --gpu-id 1
```

**Wait for:** `explorer server ready` in the terminal output.

**What it does:**
1. Starts a vLLM server for the explorer model on GPU 1, port 8011.
2. Generates 2 queries × G=2 = 4 trajectories using the explorer.
3. Sends each trajectory to the main agent (port 8010) for final answer.
4. Kills the explorer vLLM, starts the judge vLLM on GPU 1 (same port 8011).
5. Scores all 4 answers with the judge.
6. Kills the judge vLLM, restarts the explorer vLLM.
7. Pushes 4 `(trajectory, reward)` samples to `rl/trl/tmp/val_2gpu_buffer/`.
8. Loops — keeps producing samples until killed.

**Expected log lines (in order):**
```
[daemon] starting explorer vLLM (port 8011) ...
[daemon] explorer server ready
[daemon] iter=0: generating 4 trajectories
[daemon] iter=0: calling main agent ...
[daemon] swapping GPU 1 to judge mode ...
[daemon] judge server ready
[daemon] iter=0: scoring 4 answers ...
[daemon] swapping GPU 1 back to explorer ...
[daemon] iter=0: pushed 4 samples (N/4 correct)
```

**Do not proceed to Step 3 until you see** `pushed 4 samples`.

---

## Step 3 — Start the learner (Terminal 3)

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    rl/trl/grpo_train.py --config rl/trl/config_2gpu_val.yaml
```

**`--num_processes 1` disables FSDP** — the learner runs as a single process,
no changes to `grpo_train.py` needed.

**What it does:**
1. Loads Qwen3.5-0.8B + LoRA on GPU 0 (alongside the main agent vLLM).
2. Blocks at `pop_samples(..., n=4)` until the buffer has 4 samples.
3. Runs 3 GRPO training steps, logging loss and metrics each step.
4. Saves checkpoint at step 2, writes `ckpt_ready.flag`.
5. Exits after step 3; final model saved to `rl/trl/runs/val_2gpu/`.

**Expected log lines:**
```
[learner] waiting for 4 samples in buffer ...
[learner] buffer ready, starting training
[learner] step=0  loss=X.XXXX  policy_loss=...  kl_loss=...  mean_reward=...
[learner] step=1  loss=X.XXXX  ...
[learner] checkpoint saved to rl/trl/tmp/val_2gpu_buffer/ckpt_latest
[learner] step=2  loss=X.XXXX  ...
[learner] training complete, model saved to rl/trl/runs/val_2gpu/
```

---

## Step 4 — Verify results

After the learner exits cleanly, check:

```bash
# Final model exists
ls rl/trl/runs/val_2gpu/

# Buffer was drained (should be empty or near-empty)
ls rl/trl/tmp/val_2gpu_buffer/sample_*.jsonl 2>/dev/null | wc -l

# Checkpoint was written
ls rl/trl/tmp/val_2gpu_buffer/ckpt_latest/

# Daemon detected the checkpoint flag (search daemon terminal output)
# Look for: "[daemon] new checkpoint detected"
```

Also check Terminal 2 (daemon) for the flag detection log after step 2 checkpoint:
```
[daemon] new checkpoint at .../ckpt_latest, reloading ...
[daemon] explorer reloaded
```

---

## Step 5 — Cleanup and kill daemon

Once validation passes, kill the daemon and main agent (Ctrl+C in each terminal),
then clean up:

```bash
rm -rf rl/trl/tmp/val_2gpu_buffer
rm -rf rl/trl/runs/val_2gpu
```

---

## Step 6 — Submit production job

```bash
sbatch rl/trl/grpo_train.SBATCH
```

Production config is at `rl/trl/config.yaml` (unchanged throughout this validation).

---

## Checklist

- [ ] Terminal 1: main agent starts, `Application startup complete`
- [ ] Terminal 2: daemon logs `explorer server ready`
- [ ] Terminal 2: daemon logs `generating 4 trajectories`
- [ ] Terminal 2: daemon logs `calling main agent`
- [ ] Terminal 2: daemon logs `judge server ready`
- [ ] Terminal 2: daemon logs `pushed 4 samples`
- [ ] Terminal 3: learner logs `buffer ready, starting training`
- [ ] Terminal 3: learner logs `step=0`, `step=1`, `step=2` with finite loss
- [ ] Terminal 3: learner logs `training complete`
- [ ] Terminal 2: daemon logs `new checkpoint detected` after learner step 2
- [ ] `rl/trl/runs/val_2gpu/` exists and contains model files
