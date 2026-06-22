#!/usr/bin/env python3
"""Analyze the final pretokenized corpus (.pt file)."""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to .pt corpus file")
    parser.add_argument("--meta", required=True, help="Path to corpus meta JSON")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer dir")
    parser.add_argument("--samples", type=int, default=10, help="Number of random chunks to inspect")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("=" * 60)
    print("CORPUS META")
    print("=" * 60)
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nLoading corpus (memory-mapped)...")
    try:
        data = torch.load(args.pt, map_location="cpu", mmap=True)
    except TypeError:
        # Older PyTorch without mmap support
        data = torch.load(args.pt, map_location="cpu")

    if data.dim() == 1:
        seq_len = meta.get("seq_len", 512)
        data = data.view(-1, seq_len)

    chunks, seq_len = data.shape
    total_tokens = chunks * seq_len
    print("\n" + "=" * 60)
    print("SHAPE & SIZE")
    print("=" * 60)
    print(f"Chunks: {chunks:,}")
    print(f"Seq len: {seq_len}")
    print(f"Total tokens: {total_tokens:,} (~{total_tokens/1e9:.2f}B)")

    # Token ID frequency on a random subset (faster than full histogram)
    sample_size = min(100_000, chunks)
    sample_indices = random.sample(range(chunks), sample_size)
    sample_data = data[sample_indices]
    unique, counts = torch.unique(sample_data, return_counts=True)
    vocab_usage = unique.numel()
    print("\n" + "=" * 60)
    print("VOCAB USAGE (from 100k random chunks)")
    print("=" * 60)
    print(f"Unique token IDs used: {vocab_usage:,} / {len(tokenizer):,}")
    print(f"Most common token IDs: {unique[counts.argsort(descending=True)[:10]].tolist()}")
    print(f"Least used token IDs: {unique[counts.argsort()[:10]].tolist()}")

    # Special token counts
    special_ids = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in [
            ("bos", tokenizer.bos_token),
            ("eos", tokenizer.eos_token),
            ("pad", tokenizer.pad_token),
        ]
        if token is not None
    }
    print("\n" + "=" * 60)
    print("SPECIAL TOKEN COUNTS (per 100k sample chunks)")
    print("=" * 60)
    for name, tid in special_ids.items():
        cnt = (sample_data == tid).sum().item()
        print(f"{name} (id {tid}): {cnt:,}")

    # Decode random chunks
    print("\n" + "=" * 60)
    print("RANDOM DECODED SAMPLES")
    print("=" * 60)
    for i, idx in enumerate(random.sample(range(chunks), args.samples), 1):
        ids = data[idx].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"\n--- Sample {i} (chunk {idx}) ---")
        print(repr(text[:800]))


if __name__ == "__main__":
    main()
