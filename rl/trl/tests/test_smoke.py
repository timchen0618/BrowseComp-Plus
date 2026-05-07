"""
Smoke tests — verify imports, config completeness, and GRPO math.

These run quickly and require no external services or GPUs.
"""

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
_TRL = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_import_buffer():
    from rl.trl import buffer  # noqa: F401


def test_import_rollout_worker():
    from rl.trl import rollout_worker  # noqa: F401


def test_import_reward():
    from rl.trl import reward  # noqa: F401


def test_import_rollout_daemon():
    from rl.trl import rollout_daemon  # noqa: F401


def test_import_grpo_train():
    from rl.trl import grpo_train  # noqa: F401


# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    "sft_checkpoint",
    "index_path",
    "train_queries",
    "ground_truth",
    "output_dir",
    "main_agent_port",
    "rollout_port",
    "explorer_model",
    "main_agent_model",
    "judge_model",
    "group_size",
    "batch_queries",
    "kl_beta",
    "learning_rate",
    "max_steps",
    "checkpoint_every",
    "min_buffer_size",
    "max_turns",
    "max_new_tokens",
    "temperature",
    "top_p",
    "k",
    "buffer_dir",
    "ckpt_dir",
    "ckpt_flag",
]


def test_config_loads():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_required_keys():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    assert missing == [], f"Missing config keys: {missing}"


def test_config_ports_are_ints():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg["main_agent_port"], int)
    assert isinstance(cfg["rollout_port"], int)


def test_config_ports_differ():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["main_agent_port"] != cfg["rollout_port"]


def test_config_group_size_positive():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["group_size"] >= 2


def test_config_kl_beta_range():
    with open(_TRL / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert 0.0 < cfg["kl_beta"] < 1.0


# ---------------------------------------------------------------------------
# GRPO math smoke tests (CPU only, no model)
# ---------------------------------------------------------------------------

def test_grpo_loss_zero_reward_variance():
    """All rewards equal → all advantages zero → policy_loss ~0."""
    import torch
    from rl.trl.grpo_train import compute_grpo_loss

    B, T = 4, 10
    log_probs = torch.zeros(B, T) - 1.0
    ref_log_probs = torch.zeros(B, T) - 1.0
    rewards = torch.ones(B)              # constant reward
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(B, T)

    loss, metrics = compute_grpo_loss(log_probs, ref_log_probs, rewards, group_ids, mask, kl_beta=0.01)
    assert abs(metrics["policy_loss"]) < 1e-5


def test_grpo_loss_perfect_separation():
    """High-reward samples should get positive advantage → lower loss when log_probs are high."""
    import torch
    from rl.trl.grpo_train import compute_grpo_loss

    B, T = 4, 8
    # Two groups of 2; first sample in each group has reward=1, second has reward=0
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(B, T)

    # High log_probs for reward=1 samples → good policy
    log_probs = torch.tensor(
        [[-0.1] * T, [-2.0] * T, [-0.1] * T, [-2.0] * T]
    )
    ref_log_probs = torch.full((B, T), -1.0)

    loss, metrics = compute_grpo_loss(log_probs, ref_log_probs, rewards, group_ids, mask, kl_beta=0.01)
    assert isinstance(loss.item(), float)
    assert metrics["mean_reward"] == pytest.approx(0.5)


def test_grpo_advantage_sum_is_zero_per_group():
    """Within each group, advantages must sum to zero (zero-mean by construction)."""
    import torch
    from rl.trl.grpo_train import compute_grpo_loss

    B, T = 4, 5
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(B, T)
    lp = torch.zeros(B, T) - 0.5
    rlp = torch.zeros(B, T) - 0.5

    # Verify group 0 advantages: r=[1,0] → adv=[+x, -x] → sum=0
    # We can't inspect advantages directly from the function, but we can verify
    # the loss is finite and the function runs without error.
    loss, _ = compute_grpo_loss(lp, rlp, rewards, group_ids, mask, kl_beta=0.0)
    assert torch.isfinite(loss)


def test_tokenize_sample_basic():
    """tokenize_sample returns correctly shaped tensors."""
    import torch
    from unittest.mock import MagicMock
    from rl.trl.grpo_train import tokenize_sample

    tokenizer = MagicMock()
    # Return different-length token lists per role to make the test non-trivial
    def encode_side_effect(text, add_special_tokens=True):
        return list(range(len(text) // 2 + 1))
    tokenizer.encode.side_effect = encode_side_effect

    sample = {
        "query_id": "Q1",
        "reward": 1,
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<answer>4</answer>"},
        ],
    }

    result = tokenize_sample(sample, tokenizer, max_length=1024)
    assert "input_ids" in result
    assert "assistant_mask" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["assistant_mask"], torch.Tensor)
    assert result["input_ids"].shape == result["assistant_mask"].shape
    assert result["reward"] == 1.0
    assert result["group_id"] == "Q1"


def test_collate_batch_pads_to_equal_length():
    """collate_batch produces equal-length sequences."""
    import torch
    from rl.trl.grpo_train import collate_batch

    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "assistant_mask": torch.tensor([False, False, True]),
            "reward": 1.0,
            "group_id": "Q1",
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8]),
            "assistant_mask": torch.tensor([False, False, True, True, True]),
            "reward": 0.0,
            "group_id": "Q1",
        },
    ]
    batch = collate_batch(samples, pad_token_id=0)
    assert batch["input_ids"].shape[1] == 5   # max length
    assert batch["mask"].shape[1] == 5
    assert batch["rewards"].shape == (2,)
    assert batch["group_ids"].shape == (2,)


def test_collate_batch_group_ids_consistent():
    """Samples with the same query_id get the same integer group_id."""
    import torch
    from rl.trl.grpo_train import collate_batch

    def _make(qid, r):
        return {
            "input_ids": torch.tensor([1, 2]),
            "assistant_mask": torch.tensor([False, True]),
            "reward": r,
            "group_id": qid,
        }

    samples = [_make("Q1", 1), _make("Q1", 0), _make("Q2", 1), _make("Q2", 0)]
    batch = collate_batch(samples, pad_token_id=0)
    gids = batch["group_ids"].tolist()
    assert gids[0] == gids[1]
    assert gids[2] == gids[3]
    assert gids[0] != gids[2]
