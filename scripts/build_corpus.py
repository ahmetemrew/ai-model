#!/usr/bin/env python3
"""Build the pre-tokenized corpus for CPT."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import login
from transformers import AutoTokenizer

from src.data.corpus import build_corpus
from src.utils.config import get_hf_token, load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Build my-50m-model pre-training corpus")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    parser.add_argument("--sample", action="store_true", help="Process only a small sample for local testing")
    parser.add_argument("--sample-size", type=int, default=1000, help="Documents per source in sample mode")
    parser.add_argument("--output-name", default="turkish_corpus_v2", help="Output corpus base name")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    corpus_cfg = cfg["corpus"]
    seq_len = cfg["seq_len"]

    tokenizer_dir = resolve_path(cfg["tokenizer_dir"])
    output_dir = resolve_path(cfg["pretokenized_corpus"]).parent
    temp_dir = resolve_path(f"data/temp_corpus_{int(time.time())}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    token = get_hf_token()
    if token:
        print("[HF] Logging in with provided token.")
        login(token=token)
    else:
        print("[HF] No token found. Public datasets only.")

    build_corpus(
        tokenizer=tokenizer,
        corpus_cfg=corpus_cfg,
        output_dir=output_dir,
        temp_dir=temp_dir,
        seq_len=seq_len,
        output_name=args.output_name,
        sample=args.sample,
        sample_size=args.sample_size,
    )
    print("\n[Done] Corpus build finished.")


if __name__ == "__main__":
    main()
