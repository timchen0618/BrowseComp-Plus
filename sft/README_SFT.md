# Supervised Fine-Tuning (SFT) for Trajectory Following

This script trains an LM to follow agent trajectories from JSONL data.

## Data Format

Each line in the JSONL file is a JSON object with:

- **`input`**: System prompt + user prompt (string). Can be plain text or JSON array of messages.
- **`output`**: Trajectory from the agent model. Either:
  - List of messages: `[{"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, ...]`
  - String: treated as a single assistant message

**Loss masking**: Only "assistant" messages contribute to the loss; "user" messages in the trajectory are masked.

## Quick Start

```bash
# Install dependencies
pip install -r scripts/requirements-sft.txt

# Single GPU
python scripts/sft_train.py \
  --data_path scripts/sample_trajectory_data.jsonl \
  --output_dir checkpoints/sft \
  --use_lora \
  --num_epochs 2

# Multi-GPU (Accelerate)
accelerate launch scripts/sft_train.py \
  --data_path /path/to/trajectories.jsonl \
  --output_dir checkpoints/sft \
  --use_lora \
  --num_epochs 3
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | required | Path to JSONL file |
| `--model_name_or_path` | Qwen/Qwen2.5-0.5B-Instruct | Base model |
| `--output_dir` | checkpoints/sft | Checkpoint directory |
| `--use_lora` | False | Enable LoRA |
| `--validation_split` | 0.1 | 10% for validation |
| `--save_steps` | 100 | Checkpoint frequency |
| `--wandb_project` | sft-trajectory | Wandb project |

## Multi-GPU Training

Uses Hugging Face Accelerate. Configure once:

```bash
accelerate config
```

Then run with `accelerate launch`:

```bash
accelerate launch --num_processes 4 scripts/sft_train.py --data_path data.jsonl ...
```

For FSDP (Fully Sharded Data Parallel):

```bash
accelerate launch --use_fsdp scripts/sft_train.py ...
```

## Wandb

Logs `loss`, `eval_loss`, `learning_rate`, etc. Set `WANDB_API_KEY` or run `wandb login`.

## Output

- Checkpoints saved to `output_dir/checkpoint-{step}/`
- Final model at `output_dir/final/`
