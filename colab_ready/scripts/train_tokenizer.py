#!/usr/bin/env python3
"""Train a clean Turkish BPE tokenizer for my-50m-model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import login

from src.tokenizer.train import (
    build_sentencepiece_tokenizer,
    stream_training_texts,
    validate_tokenizer,
)
from src.utils.config import get_hf_token, load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Train my-50m-model tokenizer from scratch")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    parser.add_argument(
        "--output-dir",
        default="./models/my-tokenizer",
        help="Where to save the new tokenizer",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100_000_000,
        help="Total characters to collect for tokenizer training",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50176,
        help="Target vocab size (including special tokens)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Gate 1 validation (not recommended)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    corpus_cfg = cfg["corpus"]
    output_dir = resolve_path(args.output_dir)

    token = get_hf_token()
    if token:
        print("[HF] Logging in with provided token.")
        login(token=token)
    else:
        print("[HF] No token found. Public datasets will be attempted.")

    print(f"[TOKENIZER] Output dir: {output_dir}")
    print(f"[TOKENIZER] Target training chars: {args.max_chars:,}")
    print(f"[TOKENIZER] Target vocab size: {args.vocab_size}")

    # Collect texts as an iterator to avoid holding everything in memory.
    texts = stream_training_texts(corpus_cfg, max_chars=args.max_chars)

    # Train tokenizer. The iterator is consumed here.
    tokenizer = build_sentencepiece_tokenizer(
        texts,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
    )

    # Validate after saving.
    if not args.skip_validation:
        if not validate_tokenizer(tokenizer):
            print("[TOKENIZER] Validation failed.")
            sys.exit(1)

    # Quick sanity sample.
    sample = "Türkiye'nin başkenti Ankara'dır."
    ids = tokenizer.encode(sample, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"[TOKENIZER] Sample: {sample!r}")
    print(f"[TOKENIZER] IDs: {ids}")
    print(f"[TOKENIZER] Decoded: {decoded!r}")


if __name__ == "__main__":
    main()
