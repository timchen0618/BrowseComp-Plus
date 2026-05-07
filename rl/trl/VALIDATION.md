# Quick Validation Plan (no SLURM)

Run this to validate the full pipeline on live GPUs before submitting the real job.

---

## Step 1: Patch config for fast iteration

Edit `rl/trl/config.yaml`:

```yaml
explorer_model: Qwen/Qwen3.5-0.5B
main_agent_model: Qwen/Qwen3.5-0.5B
judge_model: Qwen/Qwen3.5-0.5B
batch_queries: 2
group_size: 2
min_buffer_size: 4
max_steps: 3
```

Restore original values after validation passes.

---

## Step 2: Three terminals (from project root inside Singularity)

**Terminal 1 — main agent (GPU 2)**
```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-0.5B \
    --port 8010 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
```
Wait until you see `Application startup complete`.

**Terminal 2 — rollout daemon (GPU 3)**
```bash
python rl/trl/rollout_daemon.py --config rl/trl/config.yaml --gpu-id 3
```
The daemon starts its own vLLM on GPU 3, generates trajectories, calls the main
agent, swaps to judge, scores, and pushes samples to the buffer.

**Terminal 3 — learner (GPU 0, 1)**
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
    --fsdp_sharding_strategy FULL_SHARD \
    rl/trl/grpo_train.py --config rl/trl/config.yaml
```
The learner blocks until `min_buffer_size=4` samples are in the buffer, then
runs 3 training steps and exits.

---

## Step 3: What to check

- [ ] Terminal 1: main agent server starts and answers health checks
- [ ] Terminal 2: daemon logs `explorer server ready`, then `generating N trajectories`
- [ ] Terminal 2: daemon logs `calling main agent` and `scoring N answers`
- [ ] Terminal 2: daemon logs `pushed N samples (M/N correct)`
- [ ] Terminal 3: learner logs `buffer ready, starting training`
- [ ] Terminal 3: learner logs `step=0  loss=...` for each of 3 steps
- [ ] Terminal 3: learner exits cleanly; final model saved to `rl/trl/runs/grpo_run1/`
- [ ] Buffer dir `rl/trl/tmp/rl_buffer/` is empty after learner drains it

---

## Step 4: Restore production config

```yaml
explorer_model: Qwen/Qwen3.5-4B
main_agent_model: openai/gpt-oss-120b
judge_model: Qwen/Qwen3-32B
batch_queries: 32
group_size: 4
min_buffer_size: 64
max_steps: 500
```

Then submit:
```bash
sbatch rl/trl/grpo_train.SBATCH
```
