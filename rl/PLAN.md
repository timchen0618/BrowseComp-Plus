# RL Training Plan: GRPO on Explorer Model

## Overview

Train the SFT-ed Qwen3.5-4B explorer model with GRPO, using LLM-judge accuracy
on the BrowseComp-Plus task as the reward signal.

**Models**
- Explorer (trainee): Qwen3.5-4B + LoRA (SFT checkpoint in `sft/checkpoints/`)
- Main agent (fixed): GPT-OSS-120B, served via vLLM on GPU 2
- Judge (fixed): Qwen3-32B, served via vLLM on GPU 3

**Data**
- Train: 680 queries (`topics-qrels/bcp/queries_train680.tsv`)
- Eval: 150 queries (`topics-qrels/bcp/queries_test150.tsv`)
- Ground truth: `data/browsecomp_plus_decrypted_train680.jsonl`

**Algorithm**: GRPO (Group Relative Policy Optimization)
- Group size G = 4 (try G = 8 later)
- Reward: binary {0, 1} from Qwen3-32B LLM judge
- KL reference: SFT checkpoint (frozen)

---

## GPU Layout (4× H200, single SBATCH)

| GPU | Role | Model | Notes |
|-----|------|-------|-------|
| 0, 1 | Learner | Qwen3.5-4B LoRA, FSDP TP=2 | Continuous GRPO updates |
| 2 | Main agent | GPT-OSS-120B, vLLM TP=1 | Always loaded; persistent server |
| 3 | Actor (time-muxed) | Qwen3.5-4B → Qwen3-32B | Explorer rollout, then judge; alternating |

**Memory budget (H200 = 80 GB)**
- GPU 2: ~68 GB (model) + 12 GB KV cache — `gpu_memory_utilization=0.85`
- GPU 3 (explorer): ~8 GB (model) + 72 GB KV cache
- GPU 3 (judge): ~32 GB (int8 quant) + 48 GB KV cache — use `--quantization int8`
- GPU 0,1: ~20 GB/GPU (FSDP shards + optimizer states)

---

## Architecture: Async Actor-Learner with Buffer

```
GPU 0,1  ──────────────────────────────────────────────────────────
         [  train  ][  train  ][  train  ][  train  ][  train  ]...
                ↑ pulls from filesystem buffer (blocks if empty)

GPU 2    ──────────────────────────────────────────────────────────
         [  GPT-OSS-120B vLLM — always loaded, TP=1             ]
              ↑ takes HTTP calls whenever trajectories arrive

GPU 3    ──────────────────────────────────────────────────────────
         [ explorer ][ switch ][ judge ][ switch ][ explorer ]...
              ↓ trajectories stream to GPU 2 as they complete
              ↓ scored answers pushed to buffer
```

### Rollout Iteration (GPU 3 + GPU 2)

Trajectories are streamed to the main agent as they complete — no waiting for
the full batch.

```
|── load Qwen3.5-4B (~15s) ─|
|── explorer gen (~5–10 min) ─────────────────────|
                  |── main agent calls, pipelined (~10–20 min) ──────|
                            |── kill/load judge (~30s) ──|
                                                 |── judge (~2–3 min) ──|
                                                                  └─ push to buffer
```

Net wall time ≈ `max(explorer, main_agent) + judge`

### Buffer Protocol

- Buffer lives at `tmp/rl_buffer/` as JSONL files
- Learner blocks at `min_buffer_size = 64` samples before first step
- Rollout daemon appends complete `(trajectory, reward)` pairs atomically

### Weight Sync Protocol

- Learner writes updated LoRA weights to `buffer/ckpt_latest/` every K=20 steps
- Learner creates `buffer/ckpt_ready.flag` after write completes
- Rollout daemon checks for flag at the start of each rollout iteration;
  loads new weights, deletes flag, continues

---

## Pipeline Flow

```
Query TSV
    │
    ▼
[Explorer — GPU 3, Qwen3.5-4B vLLM]
    Multi-turn generation with real FAISS search calls (Qwen3-Embedding-8B)
    G=4 trajectories per query
    │
    ▼ (stream as each trajectory completes)
[Main Agent — GPU 2, GPT-OSS-120B vLLM]
    traj_orig_ext mode (oss_client.py logic)
    Continues trajectory to final answer with search tools
    │
    ▼ (GPU 3 switches to judge)
[Judge — GPU 3, Qwen3-32B vLLM int8]
    Batch scores all G×Q final answers
    reward ∈ {0, 1}
    │
    ▼
[Buffer — filesystem JSONL]
    │
    ▼
[Learner — GPU 0,1, FSDP]
    GRPO advantages: A_i = (r_i − μ) / (σ + ε)  per query group
    Loss: −A_i · log_prob(explorer_tokens) + β · KL(policy ∥ ref_sft)
    Optimizer step → write checkpoint every K steps
```

---

## Framework

**TRL GRPOTrainer** (subclassed) + custom rollout daemon

- TRL handles: GRPO loss, KL penalty, LoRA + reference model management,
  checkpointing, logging
- We override `_generate_completions` with multi-turn tool execution loop
  (reuses tool-call logic from `tongyi_client.py`)
- Reward function: async HTTP calls to GPU 2 (main agent) + GPU 3 (judge)
- Rollout parallelism: `asyncio.gather` over G×Q concurrent rollouts

---

## Planned File Layout

```
rl/
├── PLAN.md                     ← this file
├── grpo_train.py               ← TRL GRPOTrainer subclass; learner process
├── rollout_daemon.py           ← actor loop: explorer → main agent → judge
├── rollout_worker.py           ← single multi-turn trajectory generation
├── reward.py                   ← async main agent + judge calls
├── buffer.py                   ← filesystem buffer read/write helpers
└── grpo_train.SBATCH           ← 4× H200 job: starts all processes
```

---

## Alternative Design: veRL Framework

### Key Difference from TRL Design

The TRL design's weight sync requires killing and restarting the vLLM explorer
server every K training steps (~45 s overhead). veRL eliminates this with
**in-place weight transfer**: after each training step, updated LoRA weights are
broadcast directly from the training process into the live vLLM server's GPU
memory without any restart (~2–5 s).

This comes from veRL's hybrid engine, which manages the actor model jointly
across training (FSDP) and rollout (vLLM) workers under a single Ray cluster.

### GPU Layout (same 4× H200)

| GPU | Role | Model | Notes |
|-----|------|-------|-------|
| 0, 1 | TrainerWorker | Qwen3.5-4B LoRA, FSDP TP=2 | GRPO gradient steps |
| 2 | External (not managed by veRL) | GPT-OSS-120B, vLLM TP=1 | Always loaded; same as TRL design |
| 3 | ActorRolloutRefWorker | Qwen3.5-4B (actor + frozen ref) | vLLM rollout; in-place weight sync from GPU 0,1 |

GPU 3 memory: actor weights (~8 GB) + frozen SFT ref weights (~8 GB) + KV cache
(~64 GB) = well within 80 GB.

### Architecture: Synchronous Step Loop (no buffer)

veRL runs a synchronous generate → reward → train loop per step. There is no
async buffer — the learner waits for rollout, and the rollout waits for the
updated weights before starting the next batch.

```
──── veRL step loop ────────────────────────────────────────────────────────

GPU 3   [ explorer rollout, G×Q trajs ]
             ↓ trajectories complete
GPU 2   [    main agent, pipelined    ]   ← external HTTP, concurrent
GPU 3   [ switch → judge → rewards   ]
             ↓ reward tensor
GPU 0,1 [         GRPO update        ]
             ↓ broadcast LoRA weights (~2–5 s, in-place)
GPU 3   [ next rollout (updated policy immediately) ]

────────────────────────────────────────────────────────────────────────────
```

### Weight Sync: the Core Advantage

```
TRL design (per K=20 steps):
  write ckpt to disk → rollout daemon detects flag → kill vLLM → reload → ~45 s

veRL design (every step):
  broadcast weight delta via Ray shared memory → vLLM load_weights() → ~3 s
  → policy is always on-policy; no staleness accumulation
```

### Custom Rollout Worker

veRL's `ActorRolloutRefWorker` exposes a `generate_sequences()` method that we
override to implement the multi-turn tool execution loop (same logic as
`tongyi_client.py`). This is more involved than TRL's override but provides
better infrastructure: Ray handles worker lifecycle, GPU assignment, and
fault tolerance.

```python
class ExplorerRolloutWorker(ActorRolloutRefWorker):
    def generate_sequences(self, prompts):
        trajectories = []
        for prompt in prompts:
            traj = self._run_multiturn(prompt)   # tool execution loop
            trajectories.append(traj)
        return trajectories

    def _run_multiturn(self, prompt):
        messages = [prompt]
        for _ in range(max_turns):
            out = self.vllm_engine.generate(messages, stop=["</tool_call>", "<answer>"])
            if stopped_at_answer(out):
                break
            query = parse_tool_call(out)
            result = self.searcher.search(query)
            messages.append(tool_response(result))
        return messages
```

### Reward Function

Same as TRL design: async HTTP to GPU 2 (main agent) + GPU 3 time-muxed judge.
veRL calls the reward function as a standard Python callable after rollout;
the judge GPU switch is identical.

### Planned File Layout (veRL)

```
rl/
├── PLAN.md
├── verl/
│   ├── grpo_train_verl.py          ← Ray entrypoint; veRL config + run loop
│   ├── explorer_rollout_worker.py  ← ActorRolloutRefWorker subclass
│   ├── reward.py                   ← same as TRL design (reusable)
│   └── grpo_train_verl.SBATCH      ← 4× H200; starts Ray head + workers
└── trl/
    ├── grpo_train.py
    ├── rollout_daemon.py
    ├── rollout_worker.py
    ├── reward.py
    ├── buffer.py
    └── grpo_train.SBATCH
```

---

## Design Comparison

| Aspect | TRL (async buffer) | veRL (sync step) |
|--------|-------------------|------------------|
| **Weight sync cost** | ~45 s / K steps | ~3 s / every step |
| **Policy staleness** | Up to K steps behind | Always on-policy |
| **Training while rolling out** | Yes (async buffer) | No (sync) |
| **GPU utilization GPU 0,1** | High (async, rarely idle) | Moderate (waits for rollout) |
| **Rollout throughput** | asyncio, custom daemon | Ray actors, native vLLM |
| **Multi-turn tool exec** | Override 1 method in TRL | Subclass RolloutWorker |
| **Setup complexity** | Medium (pip install trl) | High (Ray cluster, veRL install) |
| **Singularity compatibility** | Easy | Needs Ray head/worker setup |
| **Debugging** | Standard Python | Ray distributed traces |
| **Scales beyond 4 GPUs** | Moderate | Excellent |

### When veRL wins

veRL's weight sync advantage only matters if the training loop runs fast enough
that staleness actually accumulates. In our design, the bottleneck is
**main agent calls on GPU 2** (~10–20 min per batch), which dwarfs both the
training step time and the weight sync overhead. Syncing weights every step vs.
every 20 steps makes little practical difference when the policy can't change
faster than the main agent can score trajectories.

veRL becomes clearly better if:
- The main agent bottleneck is removed (e.g., a much faster reward signal)
- You scale to 8+ GPUs and need Ray's distributed coordination
- You want colocated rollout+training (veRL's sleep/wake mode) at larger scale

### Recommendation

**Start with TRL** for the current 4-GPU setup. The async buffer is a genuine
advantage given the slow main agent, the setup overhead is lower, and the
multi-turn override is simpler. **Migrate to veRL** if the project scales
beyond 4 GPUs or if the main agent bottleneck is resolved.

---

## Open Questions / Iteration Points

- [ ] Should the explorer model also call real search tools during rollout,
      or generate trajectories without actual retrieval? (Currently: real tools)
- [ ] KL coefficient β — start with 0.01, tune based on entropy collapse
- [ ] Batch queries per rollout iteration — start with 32 (→ 128 samples/iter);
      increase if learner starves
- [ ] G = 8 when GPU budget allows (more stable advantage estimates)
- [ ] Whether to add an intermediate retrieval-recall reward (Plan C hybrid)
      if binary reward is too sparse
- [ ] Eval cadence: run on test150 every N iterations using existing
      `scripts_evaluation/evaluate_run.py`
- [ ] Whether to merge LoRA into base for final checkpoint or keep adapters
