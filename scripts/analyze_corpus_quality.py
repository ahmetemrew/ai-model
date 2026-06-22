#!/usr/bin/env python3
"""Inspect data quality for each corpus source before building the full corpus."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.corpus import ENGLISH_CHARS, TURKISH_CHARS, filter_document
from src.utils.config import get_hf_token, load_yaml, resolve_path
from huggingface_hub import login


def analyze_text(text: str):
    total = len(text) or 1
    tr_count = sum(1 for c in text if c in TURKISH_CHARS)
    en_count = sum(1 for c in text if c in ENGLISH_CHARS)
    broken = text.count("�")
    words = text.split()
    unique = len(set(words))
    return {
        "length": len(text),
        "words": len(words),
        "tr_ratio": tr_count / total,
        "en_ratio": en_count / total,
        "broken_chars": broken,
        "unique_word_ratio": unique / len(words) if words else 0,
    }


def _collect_stats(texts, corpus_cfg, tokenizer, is_code, is_english, n_samples):
    passed = []
    rejected = []
    stats_passed = []
    stats_rejected = []
    for i, text in enumerate(texts):
        if i >= 2000 and len(passed) >= n_samples and len(rejected) >= n_samples:
            break
        if not text or not isinstance(text, str):
            continue
        ok = filter_document(text, corpus_cfg, is_code=is_code, is_english=is_english)
        info = analyze_text(text)
        if ok:
            if len(passed) < n_samples:
                passed.append((text, info))
            stats_passed.append(info)
        else:
            if len(rejected) < n_samples:
                rejected.append((text, info))
            stats_rejected.append(info)

    def avg(stats, key):
        return sum(s[key] for s in stats) / len(stats) if stats else 0

    print(f"\nPassed samples inspected: {len(passed)} | Rejected samples inspected: {len(rejected)}")
    print(f"Passed stats (avg): length={avg(stats_passed,'length'):.0f}, tr_ratio={avg(stats_passed,'tr_ratio'):.2f}, en_ratio={avg(stats_passed,'en_ratio'):.2f}, broken={avg(stats_passed,'broken_chars'):.1f}")
    print(f"Rejected stats (avg): length={avg(stats_rejected,'length'):.0f}, tr_ratio={avg(stats_rejected,'tr_ratio'):.2f}, en_ratio={avg(stats_rejected,'en_ratio'):.2f}, broken={avg(stats_rejected,'broken_chars'):.1f}")

    print("\n--- PASSED samples ---")
    for idx, (text, info) in enumerate(passed[:5], 1):
        print(f"\n[Passed {idx}] len={info['length']}, tr={info['tr_ratio']:.2f}, en={info['en_ratio']:.2f}, broken={info['broken_chars']}")
        print(repr(text[:600]))

    print("\n--- REJECTED samples ---")
    for idx, (text, info) in enumerate(rejected[:3], 1):
        print(f"\n[Rejected {idx}] len={info['length']}, tr={info['tr_ratio']:.2f}, en={info['en_ratio']:.2f}, broken={info['broken_chars']}")
        print(repr(text[:600]))

    print("\n--- TOKENIZE -> DECODE quality (passed) ---")
    for idx, (text, _) in enumerate(passed[:3], 1):
        ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"\n[Decoded {idx}] ids={len(ids)}")
        print(repr(decoded[:600]))


def inspect_source(name, source_cfg, tokenizer, corpus_cfg, n_samples=20):
    print(f"\n{'='*60}")
    print(f"SOURCE: {name}")
    print(f"{'='*60}")

    is_code = name == "code_stack"
    is_english = name == "fineweb_en_sample"

    if "hf_id" in source_cfg:
        hf_id = source_cfg["hf_id"]
        subset = source_cfg.get("subset")
        split = source_cfg.get("split", "train")
        streaming = source_cfg.get("streaming", True)
        text_col = source_cfg.get("text_col", "text")

        try:
            if subset:
                ds = load_dataset(hf_id, subset, split=split, streaming=streaming)
            else:
                ds = load_dataset(hf_id, split=split, streaming=streaming)
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            return

        texts = (item.get(text_col, "") for item in ds)
        _collect_stats(texts, corpus_cfg, tokenizer, is_code, is_english, n_samples)

    elif "local_jsonl" in source_cfg:
        jsonl_path = Path(source_cfg["local_jsonl"]).expanduser().resolve()
        text_col = source_cfg.get("text_col", "text")
        if not jsonl_path.exists():
            print(f"ERROR file not found: {jsonl_path}")
            return

        def gen():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get(text_col, "")
                        if isinstance(text, list):
                            text = " ".join(str(t) for t in text if isinstance(t, str))
                        yield text
                    except json.JSONDecodeError:
                        continue

        _collect_stats(gen(), corpus_cfg, tokenizer, is_code, is_english, n_samples)
    else:
        print(f"ERROR unknown source type for {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    corpus_cfg = cfg["corpus"]
    tokenizer_dir = resolve_path(cfg["tokenizer_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    token = get_hf_token()
    if token:
        login(token=token)

    for name, source_cfg in corpus_cfg["sources"].items():
        inspect_source(name, source_cfg, tokenizer, corpus_cfg, n_samples=args.samples)


if __name__ == "__main__":
    main()
