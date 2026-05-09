"""
Tier-1 CPU pipeline tests.

Tests the full tokenize → collate → compute_log_probs → compute_grpo_loss
path with mocked models (no GPU, no real LLM needed).  Also validates the
validation config and the stale-date fix on _explorer_system_prompt().
"""

import sys
import yaml
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_ROOT = Path(__file__).parent.parent.parent.parent
_TRL = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# _explorer_system_prompt: date is fresh (not frozen at import time)
# ---------------------------------------------------------------------------

def test_explorer_system_prompt_has_today():
    from rl.trl.rollout_worker import _explorer_system_prompt
    assert date.today().isoformat() in _explorer_system_prompt()


def test_explorer_system_prompt_not_stale():
    """Two calls on the same day return the same date (and it is today's)."""
    from rl.trl.rollout_worker import _explorer_system_prompt
    p1, p2 = _explorer_system_prompt(), _explorer_system_prompt()
    assert date.today().isoformat() in p1
    assert p1 == p2


# ---------------------------------------------------------------------------
# config_2gpu_val.yaml correctness
# ---------------------------------------------------------------------------

def test_val_config_loads():
    with open(_TRL / "config_2gpu_val.yaml") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_val_config_batch_fills_buffer():
    """group_size * batch_queries must equal min_buffer_size."""
    with open(_TRL / "config_2gpu_val.yaml") as f:
        cfg = yaml.safe_load(f)
    produced = cfg["group_size"] * cfg["batch_queries"]
    assert produced == cfg["min_buffer_size"], (
        f"group_size({cfg['group_size']}) * batch_queries({cfg['batch_queries']}) "
        f"= {produced} != min_buffer_size({cfg['min_buffer_size']})"
    )


def test_val_config_max_steps_small():
    with open(_TRL / "config_2gpu_val.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["max_steps"] <= 5, "val config should use a small max_steps"


def test_val_config_ports_differ():
    with open(_TRL / "config_2gpu_val.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["main_agent_port"] != cfg["rollout_port"]


# ---------------------------------------------------------------------------
# compute_log_probs: shape regression test (the [B,T] × [B,T-1] crash)
# ---------------------------------------------------------------------------

def _make_mock_model(B: int, T: int, V: int, training: bool = True) -> MagicMock:
    model = MagicMock()
    model.training = training
    model.return_value.logits = torch.randn(B, T, V)
    return model


def test_compute_log_probs_output_shape():
    """compute_log_probs must return [B, T], not [B, T-1]."""
    from rl.trl.grpo_train import compute_log_probs
    B, T, V = 4, 20, 512
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T)
    mask[:, T // 2:] = 1.0
    out = compute_log_probs(_make_mock_model(B, T, V), input_ids, mask)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


def test_compute_log_probs_zeros_on_non_assistant():
    """Positions where mask=0 must be zero in the output."""
    from rl.trl.grpo_train import compute_log_probs
    B, T, V = 2, 10, 100
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T)
    mask[:, 5:] = 1.0
    out = compute_log_probs(_make_mock_model(B, T, V), input_ids, mask)
    assert (out[:, :5] == 0.0).all(), "non-assistant positions should be zero"


def test_compute_log_probs_nonzero_on_assistant():
    """At least some assistant positions should be non-zero (sanity)."""
    from rl.trl.grpo_train import compute_log_probs
    B, T, V = 2, 10, 100
    torch.manual_seed(0)
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T)
    mask[:, 3:] = 1.0
    out = compute_log_probs(_make_mock_model(B, T, V), input_ids, mask)
    assert (out[:, 3:] != 0.0).any(), "at least some assistant positions should be non-zero"


# ---------------------------------------------------------------------------
# Full CPU pipeline: tokenize → collate → log_probs → grpo_loss
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab_size: int = 256) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    def _encode(text, add_special_tokens=True):
        return [i % vocab_size for i in range(max(1, len(text) // 3))]
    tok.encode.side_effect = _encode
    return tok


def _raw_sample(query_id: str, reward: int) -> dict:
    return {
        "query_id": query_id,
        "reward": reward,
        "messages": [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"France capital\"}}</tool_call>"},
            {"role": "user", "content": "<tool_response>[{\"docid\":\"d1\",\"text\":\"Paris\"}]</tool_response>"},
            {"role": "assistant", "content": "<answer>Paris</answer>"},
        ],
    }


def test_full_cpu_pipeline_shapes():
    """tokenize → collate → log_probs → grpo_loss all produce finite tensors."""
    from rl.trl.grpo_train import (
        tokenize_sample, collate_batch, compute_log_probs, compute_grpo_loss,
    )
    V = 256
    tokenizer = _make_tokenizer(V)

    raw = [_raw_sample("Q1", 1), _raw_sample("Q1", 0),
           _raw_sample("Q2", 1), _raw_sample("Q2", 0)]
    tokenised = [tokenize_sample(s, tokenizer, max_length=512) for s in raw]
    batch = collate_batch(tokenised, pad_token_id=0)

    B = len(raw)
    T = batch["input_ids"].shape[1]

    policy = _make_mock_model(B, T, V, training=True)
    ref    = _make_mock_model(B, T, V, training=False)

    lp  = compute_log_probs(policy, batch["input_ids"], batch["mask"])
    rlp = compute_log_probs(ref,    batch["input_ids"], batch["mask"])

    assert lp.shape  == (B, T)
    assert rlp.shape == (B, T)

    loss, metrics = compute_grpo_loss(
        lp, rlp, batch["rewards"], batch["group_ids"], batch["mask"], kl_beta=0.01,
    )
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    for k, v in metrics.items():
        assert torch.isfinite(torch.tensor(v)), f"metric {k}={v} is not finite"


def test_full_cpu_pipeline_all_same_reward():
    """With uniform reward, advantages are zero → policy_loss is ~0."""
    from rl.trl.grpo_train import (
        tokenize_sample, collate_batch, compute_log_probs, compute_grpo_loss,
    )
    V = 64
    tokenizer = _make_tokenizer(V)
    raw = [_raw_sample("Q1", 1), _raw_sample("Q1", 1)]
    tokenised = [tokenize_sample(s, tokenizer) for s in raw]
    batch = collate_batch(tokenised, pad_token_id=0)
    B, T = batch["input_ids"].shape

    lp  = compute_log_probs(_make_mock_model(B, T, V), batch["input_ids"], batch["mask"])
    rlp = compute_log_probs(_make_mock_model(B, T, V), batch["input_ids"], batch["mask"])

    _, metrics = compute_grpo_loss(
        lp, rlp, batch["rewards"], batch["group_ids"], batch["mask"], kl_beta=0.0,
    )
    assert abs(metrics["policy_loss"]) < 1e-4, (
        f"policy_loss should be ~0 with uniform reward, got {metrics['policy_loss']}"
    )
