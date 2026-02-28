#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for training an LM to follow agent trajectories.

Data format: JSONL file with "input" and "output" fields.
- input: system prompt + user prompt (string)
- output: trajectory from agent model - list of messages with "role" (assistant/user) and "content"

Loss is computed only on "assistant" messages; "user" messages in the trajectory are masked.

Supports:
- LoRA (optional) for parameter-efficient fine-tuning
- Multi-GPU training via Hugging Face Accelerate (FSDP/DP)
- Wandb logging
- Checkpoint saving
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

try:
    from accelerate.state import PartialState
    from accelerate.utils import DistributedType
except ImportError:
    PartialState = None
    DistributedType = None


def load_jsonl_trajectories(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file with input/output format."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    return data


def parse_output_to_messages(output: Any) -> List[Dict[str, str]]:
    """
    Parse the output field into a list of messages with role and content.
    Supports:
    - List of dicts: [{"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}]
    - String: treated as single assistant message
    """
    if isinstance(output, list):
        messages = []
        for msg in output:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"].lower()
                if role in ("assistant", "user", "system"):
                    messages.append({"role": role, "content": str(msg["content"])})
            elif isinstance(msg, dict) and "content" in msg:
                # Assume assistant if no role
                messages.append({"role": "assistant", "content": str(msg["content"])})
        return messages
    elif isinstance(output, str):
        return [{"role": "assistant", "content": output}]
    else:
        raise ValueError(f"Unsupported output type: {type(output)}")


def parse_input_to_messages(input_str: str) -> List[Dict[str, str]]:
    """
    Parse input (system prompt + user prompt) into messages.
    If input contains explicit structure (e.g., JSON), parse it.
    Otherwise treat as a single user message (system+user concatenated).
    """
    input_str = str(input_str).strip()
    # Try to parse as JSON (e.g., [{"role":"system","content":"..."},{"role":"user","content":"..."}])
    try:
        parsed = json.loads(input_str)
        if isinstance(parsed, list):
            messages = []
            for msg in parsed:
                if isinstance(msg, dict) and "content" in msg:
                    role = msg.get("role", "user").lower()
                    if role in ("system", "user", "assistant"):
                        messages.append({"role": role, "content": str(msg["content"])})
            if messages:
                return messages
    except (json.JSONDecodeError, TypeError):
        pass

    # Default: treat entire input as user message (system+user concatenated)
    return [{"role": "user", "content": input_str}]


def build_messages(input_str: str, output: Any) -> List[Dict[str, str]]:
    """Build full conversation: input messages + output trajectory messages."""
    input_messages = parse_input_to_messages(input_str)
    output_messages = parse_output_to_messages(output)
    return input_messages + output_messages


def build_assistant_mask(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
) -> List[bool]:
    """
    Build a per-token mask: True for assistant tokens (include in loss), False for others (mask).
    Finds assistant content spans in the chat-template-rendered text via regex.
    """
    import re

    if not messages:
        return []

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_offsets = encoding["offset_mapping"]
    mask = [False] * len(token_offsets)

    # Find all assistant content spans: <|im_start|>assistant\n...<|im_end|>
    # Match from assistant start to im_end (content is between \n and <|im_end|>)
    pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    for m in re.finditer(pattern, full_text, re.DOTALL):
        start_char, end_char = m.start(1), m.end(1)  # content only, not the tags
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_start < end_char and tok_end > start_char:
                mask[i] = True

    return mask


def create_dataset(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    add_assistant_masks: bool = True,
) -> Dataset:
    """Convert loaded data into HuggingFace Dataset with 'messages' format."""
    records = []
    for item in data:
        if "input" not in item or "output" not in item:
            print(f"Skipping item missing 'input' or 'output': {list(item.keys())}")
            continue
        messages = build_messages(item["input"], item["output"])
        if not messages:
            continue
        record = {"messages": messages}
        if add_assistant_masks:
            try:
                mask = build_assistant_mask(messages, tokenizer)
                # TRL DataCollatorForLanguageModeling expects "assistant_masks" (plural)
                record["assistant_masks"] = [1 if m else 0 for m in mask]
            except Exception as e:
                print(f"Warning: Could not build assistant mask, using full loss: {e}")
                # Fallback: no masking (loss on all tokens)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                record["assistant_masks"] = [1] * n_tokens
        records.append(record)
    return Dataset.from_list(records)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """
    Pre-tokenize dataset so input_ids and assistant_masks align.
    Required for assistant-only loss with TRL's DataCollatorForLanguageModeling.
    """

    def tokenize_fn(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "assistant_masks" in example:
            mask = example["assistant_masks"]
            # Truncate mask to match token length
            out["assistant_masks"] = mask[: len(enc["input_ids"])]
        return out

    return dataset.map(
        tokenize_fn,
        remove_columns=["messages", "assistant_masks"] if "assistant_masks" in dataset.column_names else ["messages"],
        desc="Tokenizing",
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """Create LoRA config. target_modules can be auto-detected for common architectures."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,  # None = auto for many models (Qwen, Llama, etc.)
    )


def main():
    parser = argparse.ArgumentParser(description="SFT training for trajectory following")
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to JSONL file with 'input' and 'output' fields",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints/sft"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="LoRA target modules (e.g., q_proj v_proj). Default: auto",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default 10%%)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sft-trajectory",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto from output_dir)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 instead of bf16",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--force_gradient_checkpointing_bf16",
        action="store_true",
        help="Force gradient checkpointing with bf16 (may cause CheckpointError)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit for memory efficiency (QLoRA)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    # Resolve bf16/fp16
    use_bf16 = args.bf16 and not args.fp16

    # FSDP uses its own activation checkpointing; gradient_checkpointing in TrainingArguments must be False
    # Check env var first (set by "accelerate launch --use_fsdp") - PartialState may report MULTI_GPU until model is wrapped
    is_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    if not is_fsdp and PartialState is not None and DistributedType is not None:
        try:
            state = PartialState()
            is_fsdp = state.distributed_type == DistributedType.FSDP
        except Exception:
            pass
    # bf16 + gradient_checkpointing causes CheckpointError (bf16/float32 metadata mismatch) - disable ckpt for bf16
    use_gradient_checkpointing = (
        args.gradient_checkpointing
        and not is_fsdp
        and (not use_bf16 or args.force_gradient_checkpointing_bf16)
    )

    # Load data
    print(f"Loading data from {args.data_path}")
    raw_data = load_jsonl_trajectories(args.data_path)
    print(f"Loaded {len(raw_data)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    full_dataset = create_dataset(raw_data, tokenizer)
    print(f"Created dataset with {len(full_dataset)} examples")

    # Train/val split
    split = full_dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Pre-tokenize for assistant-only loss masking (aligns assistant_masks with input_ids)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_seq_length)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args.max_seq_length)

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )


    peft_config = None
    if args.use_lora:
        peft_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Data collator for assistant-only loss (uses assistant_masks from pre-tokenized data)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    if use_gradient_checkpointing:
        # use_reentrant=False + determinism_check="none" to avoid bf16/float32 metadata mismatch
        # (recompute can produce float32 under autocast while forward used bf16)
        _ckpt_kwargs = {"use_reentrant": False, "determinism_check": "none"}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=_ckpt_kwargs)
    print('use gradient checkpointing:', use_gradient_checkpointing)
    # Training config
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        bf16=use_bf16,
        fp16=args.fp16,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False, "determinism_check": "none"} if use_gradient_checkpointing else None,
        report_to="wandb",
        run_name=args.wandb_run_name or args.output_dir.name,
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
    )

    # Initialize wandb
    os.environ["WANDB_PROJECT"] = args.wandb_project

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # PEFT+FSDP: use_orig_params must be false; override auto_wrap_policy for LoRA
    if args.use_lora and getattr(trainer.accelerator.state, "fsdp_plugin", None) is not None:
        from peft.utils.other import fsdp_auto_wrap_policy
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(
            trainer.model
        )

    trainer.train()
    # FSDP: use FULL_STATE_DICT for save_model to get loadable adapter weights
    if getattr(trainer, "is_fsdp_enabled", False):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))

    print(f"Training complete. Model saved to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
