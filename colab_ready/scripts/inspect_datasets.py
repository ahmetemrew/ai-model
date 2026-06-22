#!/usr/bin/env python3
"""Inspect a few rows from each configured corpus source."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.utils.config import get_hf_token, load_yaml
from huggingface_hub import login


SOURCES = {
    "fineweb2_tr": ("HuggingFaceFW/fineweb-2", "tur_Latn", "train", True, "text"),
    "cosmos_corpus": ("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", None, "train", True, "text"),
    "wikipedia_tr": ("wikipedia", "20240601.tr", "train", False, "text"),
    "fineweb_en_sample": ("HuggingFaceFW/fineweb", "CC-MAIN-2024-10", "train", True, "text"),
    "code_stack": ("bigcode/the-stack-dedup", "Python", "train", True, "content"),
}


def inspect(name, hf_id, subset, split, streaming, text_col, n=3):
    print(f"\n=== {name} ({hf_id}, subset={subset}) ===")
    try:
        if subset:
            ds = load_dataset(hf_id, subset, split=split, streaming=streaming)
        else:
            ds = load_dataset(hf_id, split=split, streaming=streaming)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    for i, row in enumerate(ds):
        if i >= n:
            break
        print(f"--- row {i} ---")
        print("columns:", list(row.keys()))
        text = row.get(text_col, "")
        print("text preview:", repr(text[:300]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", help="Source name or 'all'")
    args = parser.parse_args()

    token = get_hf_token()
    if token:
        login(token=token)

    if args.source == "all":
        for name, (hf_id, subset, split, streaming, text_col) in SOURCES.items():
            inspect(name, hf_id, subset, split, streaming, text_col)
    else:
        info = SOURCES.get(args.source)
        if not info:
            print(f"Unknown source: {args.source}")
            return
        inspect(args.source, *info)


if __name__ == "__main__":
    main()
