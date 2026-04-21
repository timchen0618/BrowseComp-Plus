#!/usr/bin/env python3
"""
Supervised fine-tuning with Hugging Face Trainer on Axolotl-style JSONL.

Expects each line: {"messages": [{"role": "...", "content": "..."}, ...]}
as produced by sft/axolotl/prepare_dataset.py.

Requires: torch, transformers (>=4.44 recommended for assistant token masks),
datasets, accelerate, peft.

Some conversations end with a user/tool turn; Axolotl warns that the last turn
is then not trainable. Assistant-only loss still applies to prior assistant
spans.

Usage (after preparing data):
  accelerate launch sft/hf_trainer/train_sft.py \\
      --model_name_or_path Qwen/Qwen3-30B-A3B \\
      --train_file sft/hf_trainer/data/train.jsonl \\
      --eval_file sft/hf_trainer/data/val.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def _apply_chat_template_with_assistant_mask(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_length: int,
) -> Dict[str, Any]:
    """Call apply_chat_template with assistant mask; support minor API variants."""
    base_kw: Dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "add_generation_prompt": False,
        "truncation": True,
        "max_length": max_length,
    }
    # Newer transformers: return_assistant_tokens_mask + assistant_masks
    try:
        return tokenizer.apply_chat_template(
            messages,
            **base_kw,
            return_assistant_tokens_mask=True,
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            messages,
            **base_kw,
            return_assistant_mask=True,
        )
    except TypeError:
        sig = inspect.signature(tokenizer.apply_chat_template)
        raise RuntimeError(
            "This tokenizer does not support assistant token masks in apply_chat_template. "
            "Upgrade transformers to a version that supports "
            "return_assistant_tokens_mask / return_assistant_mask. "
            f"Supported parameters: {list(sig.parameters.keys())}"
        ) from None


def _extract_assistant_mask(out: Dict[str, Any]) -> Optional[Sequence[Any]]:
    for key in ("assistant_masks", "assistant_mask"):
        if key in out and out[key] is not None:
            return out[key]
    return None


def _build_labels(input_ids: List[int], mask: Optional[Sequence[Any]]) -> List[int]:
    if mask is None:
        raise ValueError("Missing assistant mask in tokenizer output.")
    if len(mask) != len(input_ids):
        raise ValueError(
            f"assistant mask length {len(mask)} != input_ids length {len(input_ids)}"
        )
    labels: List[int] = []
    for tid, m in zip(input_ids, mask):
        train = bool(m) if not isinstance(m, (int, float)) else bool(int(m))
        labels.append(int(tid) if train else -100)
    return labels


def _tokenize_batch(
    tokenizer: Any,
    examples: Dict[str, List],
    max_length: int,
) -> Dict[str, List]:
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    for messages in examples["messages"]:
        out = _apply_chat_template_with_assistant_mask(
            tokenizer, messages, max_length=max_length
        )
        ids = out["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        mask = _extract_assistant_mask(out)
        labels = _build_labels(ids, mask)
        input_ids_list.append(ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}


def _lora_config(args: argparse.Namespace) -> LoraConfig:
    """LoRA aligned with sft/axolotl/qwen3_30b_a3b_search_sft.yaml defaults."""
    kwargs: Dict[str, Any] = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    # Match Axolotl lora_target_linear: true — prefer all-linear when supported.
    try:
        return LoraConfig(target_modules="all-linear", **kwargs)
    except (TypeError, ValueError):
        kwargs["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        return LoraConfig(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Base model (default matches Axolotl Qwen3-30B-A3B config).",
    )
    p.add_argument(
        "--train_file",
        type=Path,
        default=Path("sft/hf_trainer/data/train.jsonl"),
        help="Training JSONL with a `messages` column per line.",
    )
    p.add_argument(
        "--eval_file",
        type=Path,
        default=None,
        help="Optional validation JSONL (same schema). If omitted, eval is disabled.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sft/checkpoints/hf-trainer-qwen3-30b-a3b"),
        help="Where to write checkpoints and the final adapter.",
    )
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable when using FSDP activation checkpointing to avoid double checkpointing.",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true", help="Full fine-tune (no LoRA).")
    # FSDP (mirrors Axolotl yaml knobs where possible)
    p.add_argument(
        "--fsdp",
        type=str,
        default="",
        help='Comma-separated FSDP policy, e.g. "full_shard,auto_wrap". Empty disables FSDP.',
    )
    p.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default="Qwen3MoeDecoderLayer",
        help="Decoder layer class name for auto_wrap (Qwen3 MoE).",
    )
    p.add_argument(
        "--fsdp_use_orig_params",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must be false for LoRA + FSDP.",
    )
    p.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="e.g. wandb or none",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default="hf-trainer-sft",
        help="Used for W&B run name when report_to=wandb.",
    )
    p.add_argument(
        "--map_num_proc",
        type=int,
        default=1,
        help="Processes for datasets.map tokenization (1 avoids tokenizer pickling issues).",
    )
    return p.parse_args()


def _load_raw_datasets(args: argparse.Namespace) -> DatasetDict:
    if not args.train_file.is_file():
        sys.exit(f"train_file not found: {args.train_file}")
    data_files: Dict[str, str] = {"train": str(args.train_file)}
    if args.eval_file is not None and args.eval_file.is_file():
        data_files["validation"] = str(args.eval_file)
    return load_dataset("json", data_files=data_files)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    raw = _load_raw_datasets(args)
    if "messages" not in raw["train"].column_names:
        sys.exit(
            "Expected a `messages` column in train JSONL (Axolotl chat_template format)."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Smoke-test assistant masks before mapping the full dataset
    probe = _apply_chat_template_with_assistant_mask(
        tokenizer, raw["train"][0]["messages"], max_length=args.max_seq_length
    )
    if _extract_assistant_mask(probe) is None:
        sys.exit(
            "Tokenizer output has no assistant_masks / assistant_mask. "
            "Upgrade transformers or use a chat template that supports assistant masks."
        )

    tokenize_fn = partial(_tokenize_batch, tokenizer, max_length=args.max_seq_length)
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw["train"].column_names,
        num_proc=args.map_num_proc,
    )

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if not args.no_lora:
        model = get_peft_model(model, _lora_config(args))

    if args.gradient_checkpointing:
        if not args.no_lora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    has_eval = "validation" in tokenized
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "seed": args.seed,
        "eval_strategy": "steps" if has_eval else "no",
        "save_strategy": "steps",
        "load_best_model_at_end": has_eval,
        "greater_is_better": False,
        "remove_unused_columns": False,
    }
    if has_eval:
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["metric_for_best_model"] = "eval_loss"
    if args.fsdp.strip():
        parts = [x.strip() for x in args.fsdp.split(",") if x.strip()]
        training_kwargs["fsdp"] = parts
        training_kwargs["fsdp_transformer_layer_cls_to_wrap"] = (
            args.fsdp_transformer_layer_cls_to_wrap
        )
        training_kwargs["fsdp_use_orig_params"] = args.fsdp_use_orig_params

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
