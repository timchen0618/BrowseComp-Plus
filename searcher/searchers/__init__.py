"""
Searchers package for different search implementations.

All concrete searcher imports are lazy to avoid pulling in heavy dependencies
(Java for BM25, faiss/tevatron for FAISS) at module load time.
"""

from enum import Enum

from .base import BaseSearcher

# Map cli_name -> (module_name, class_name) for lazy import
_LAZY_SEARCHER_MAP = {
    "bm25": (".bm25_searcher", "BM25Searcher"),
    "faiss": (".faiss_searcher", "FaissSearcher"),
    "reasonir": (".faiss_searcher", "ReasonIrSearcher"),
    "custom": (".custom_searcher", "CustomSearcher"),
}


def _lazy_import(cli_name):
    if cli_name not in _LAZY_SEARCHER_MAP:
        raise ValueError(f"Unknown searcher type: {cli_name}")
    module_name, class_name = _LAZY_SEARCHER_MAP[cli_name]
    import importlib
    mod = importlib.import_module(module_name, package=__package__)
    return getattr(mod, class_name)


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = "bm25"
    FAISS = "faiss"
    REASONIR = "reasonir"
    CUSTOM = "custom"

    def __init__(self, cli_name):
        self.cli_name = cli_name

    @property
    def searcher_class(self):
        return _lazy_import(self.cli_name)

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
