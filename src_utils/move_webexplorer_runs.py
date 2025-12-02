#!/usr/bin/env python3

"""Move WebExplorer run files into a dedicated directory.

This script scans all JSON files in `runs/bm25/tongyi` and relocates the files
belonging to the `hkust-nlp/WebExplorer-8B` model into `runs/bm25/webexplorer`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "hkust-nlp/WebExplorer-8B"
DEFAULT_SOURCE_DIR = Path("runs/bm25/tongyi")
DEFAULT_TARGET_DIR = Path("runs/bm25/webexplorer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the source JSON files. Default: %(default)s",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory where matching files will be moved. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model name to filter on. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def load_model_name(json_path: Path) -> str | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as exc:
        logging.warning("Skipping %s (invalid JSON): %s", json_path, exc)
        return None
    except OSError as exc:
        logging.warning("Skipping %s (unable to read): %s", json_path, exc)
        return None

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        logging.debug("Missing metadata in %s", json_path)
        return None

    model_name = metadata.get("model")
    if not isinstance(model_name, str):
        logging.debug("Missing model name in %s", json_path)
        return None

    return model_name


def move_matching_files(
    source_dir: Path,
    target_dir: Path,
    target_model: str,
    dry_run: bool,
) -> int:
    if not source_dir.is_dir():
        logging.error("Source directory does not exist: %s", source_dir)
        return 1

    moved = 0
    for json_path in sorted(source_dir.glob("*.json")):
        model_name = load_model_name(json_path)
        if model_name != target_model:
            continue

        destination = target_dir / json_path.name
        target_dir.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            logging.warning("Destination already has %s; skipping.", destination)
            continue

        if dry_run:
            logging.info("[dry-run] Would move %s -> %s", json_path, destination)
        else:
            logging.info("Moving %s -> %s", json_path, destination)
            shutil.move(str(json_path), str(destination))

        moved += 1

    if moved == 0:
        logging.info("No files matched model %r in %s.", target_model, source_dir)
    else:
        logging.info("Moved %d file(s) matching model %r.", moved, target_model)

    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    return move_matching_files(
        source_dir=args.source,
        target_dir=args.target,
        target_model=args.model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())


