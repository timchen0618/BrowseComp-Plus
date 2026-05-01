"""BGE-M3 searcher adapter — wraps frames/retrieval/bge_m3_backend.py for the
BCP SearcherType registry.

The upstream class lives in the parent project (frames/retrieval/) and uses a
keyword-arg constructor. BCP clients call `searcher_class(args)` with an argparse
Namespace, so this module bridges the gap.
"""

import sys
from pathlib import Path

# frames/retrieval lives outside the BCP repo; inject onto sys.path lazily.
_FRAMES_RETRIEVAL = Path(__file__).resolve().parents[4] / "frames" / "retrieval"
if str(_FRAMES_RETRIEVAL) not in sys.path:
    sys.path.insert(0, str(_FRAMES_RETRIEVAL))

from bge_m3_backend import BgeM3Searcher as _BgeM3Backend  # noqa: E402


class BgeM3Searcher(_BgeM3Backend):
    """BCP-compatible adapter: accepts an argparse Namespace in __init__."""

    @classmethod
    def parse_args(cls, parser):
        return _BgeM3Backend.parse_args(parser)

    def __init__(self, args):
        super().__init__(
            index_path=args.bge_index_path,
            texts_path=args.bge_texts_path,
            nprobe=getattr(args, "bge_nprobe", 128),
            device=getattr(args, "bge_device", "cpu"),
        )

    def search_description(self, k: int = 10) -> str:
        # BCP BaseSearcher signature passes k; parent's signature takes no args.
        return (
            f"Search Wikipedia (June 2024 snapshot) using BGE-M3 dense retrieval. "
            f"Returns the top {k} relevant passages with their source URLs."
        )
