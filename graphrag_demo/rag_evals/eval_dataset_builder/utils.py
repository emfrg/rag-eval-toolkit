"""Utility functions for file operations and logging."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any


def write_jsonl(filepath: Path, data: List[Dict[str, Any]]) -> None:
    """Write data to JSONL file."""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")


def read_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Read data from JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
