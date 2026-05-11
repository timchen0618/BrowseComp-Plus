import sys, os
from unittest.mock import MagicMock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "search_agent"))

# Stub heavy optional dependencies that are not available in all test environments
for _mod in ("openai", "transformers", "pyserini", "pyserini.search",
             "pyserini.search.lucene", "faiss", "torch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Stub searcher subpackage so SearcherType import succeeds
if "searcher" not in sys.modules:
    sys.modules["searcher"] = MagicMock()
if "searcher.searchers" not in sys.modules:
    _ss = MagicMock()
    _ss.SearcherType = MagicMock()
    sys.modules["searcher.searchers"] = _ss

from search_agent.oss_client import (
    TRAJ_TRIGGERS,
    _build_continuation_messages,
)
from search_agent.prompts import QUERY_TEMPLATE_NO_GET_DOCUMENT


def _make_args(query_template="QUERY_TEMPLATE_NO_GET_DOCUMENT", verbose=False):
    import types
    return types.SimpleNamespace(query_template=query_template, verbose=verbose)


SAMPLE_ORIGINAL_MESSAGES = [
    {"role": "user", "content": "You are a deep research agent. Budget 5.\n\nQuestion: What is X?"},
    {"type": "reasoning", "id": "rs_001", "content": [{"text": "I need to find X."}]},
    {"type": "function_call", "name": "search", "arguments": '{"user_query": "X"}', "call_id": "c1"},
    {"type": "function_call_output", "call_id": "c1", "output": '[{"docid": "d1", "score": 0.9}]'},
]

SAMPLE_TRAJ = {
    "query_id": "Q1",
    "original_messages": SAMPLE_ORIGINAL_MESSAGES,
    "result": [],
    "status": "incomplete",
}


def test_traj_continue_in_traj_triggers():
    assert "traj_continue" in TRAJ_TRIGGERS


def test_build_continuation_messages_replaces_first_message():
    args = _make_args()
    trajectories = {"Q1": SAMPLE_TRAJ}
    msgs = _build_continuation_messages("Q1", "What is X?", args, trajectories)
    assert msgs[0]["role"] == "user"
    assert "What is X?" in msgs[0]["content"]
    assert "Budget 5" not in msgs[0]["content"]
    assert msgs[1:] == SAMPLE_ORIGINAL_MESSAGES[1:]
    assert len(msgs) == len(SAMPLE_ORIGINAL_MESSAGES)


def test_build_continuation_messages_missing_original_messages():
    args = _make_args()
    trajectories = {"Q1": {"query_id": "Q1", "result": []}}
    msgs = _build_continuation_messages("Q1", "What is X?", args, trajectories)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "What is X?" in msgs[0]["content"]


def test_build_continuation_messages_missing_query_id():
    args = _make_args()
    msgs = _build_continuation_messages("Q_UNKNOWN", "What is X?", args, {})
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"

def test_parse_run_name_traj_continue():
    from submit_missing import parse_run_name
    model, mode, seed, traj_model = parse_run_name(
        "gpt-oss-120b_traj_continue_gpt-oss-120b_seed0"
    )
    assert model == "gpt-oss-120b"
    assert mode == "traj_continue"
    assert seed == 0
    assert traj_model == "gpt-oss-120b"


def test_parse_run_name_traj_continue_cross_model():
    from submit_missing import parse_run_name
    model, mode, seed, traj_model = parse_run_name(
        "gpt-oss-120b_traj_continue_qwen3.5-4b_seed1"
    )
    assert model == "gpt-oss-120b"
    assert mode == "traj_continue"
    assert seed == 1
    assert traj_model == "qwen3.5-4b"
