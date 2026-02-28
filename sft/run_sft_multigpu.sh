#!/bin/bash
# Multi-GPU SFT training script

# Uses Hugging Face Accelerate for distributed training.
# For 1 GPU: python scripts/sft_train.py ...
# For multi-GPU: accelerate launch scripts/sft_train.py ...

# Configure accelerate first (one-time):
#   accelerate config
# Select: multi-GPU, FSDP or DeepSpeed as needed

# Example: 4 GPUs with FSDP
# Use fsdp_config.yaml (fsdp_use_orig_params=false) for LoRA+FSDP compatibility
accelerate launch \
    --config_file sft/fsdp_config.yaml \
    sft/sft_train.py \
    --gradient_checkpointing \
    --bf16 \
    --data_path sft/sft_trajectories.jsonl \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir sft/checkpoints/sft_run \
    --use_lora \
    --num_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --eval_steps 100 \
    --wandb_project sft-trajectory \
    "$@"
