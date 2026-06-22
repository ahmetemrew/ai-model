import gc
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset


TURKISH_CHARS = set("abcçdefgğhıijklmnoöprsştuüvyzâîû")
TURKISH_CHARS_UPPER = set("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZÂÎÛ")
ENGLISH_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def is_turkish_char(c: str) -> bool:
    return c in TURKISH_CHARS or c in TURKISH_CHARS_UPPER


def filter_document(
    text: str,
    corpus_cfg: Dict[str, Any],
    is_code: bool = False,
    is_english: bool = False,
) -> bool:
    if not text or not isinstance(text, str):
        return False

    filters = corpus_cfg["filters"]
    if len(text) < filters["min_chars"] or len(text) > filters["max_chars"]:
        return False

    total_chars = len(text)

    if not is_code:
        if not is_english:
            # Turkish sources should contain enough Turkish characters.
            turkish_count = sum(1 for c in text if is_turkish_char(c))
            if total_chars > 0 and turkish_count / total_chars < filters["turkish_char_ratio"]:
                return False
        else:
            # English sources should contain enough English characters.
            english_count = sum(1 for c in text if c in ENGLISH_CHARS)
            if total_chars > 0 and english_count / total_chars < filters["english_char_ratio"]:
                return False

    if not is_code:
        words = text.split()
        if len(words) < 4:
            return False

        ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        if ngrams:
            ngram_counts = Counter(ngrams)
            max_repeat = max(ngram_counts.values()) / len(ngrams)
            if max_repeat > filters["max_repetition_ratio"]:
                return False

        unique_words = len(set(words))
        if len(words) > 0 and unique_words / len(words) < filters["min_type_token_ratio"]:
            return False

    if re.search(r"\b(lorem ipsum|javascript:|function\s*\(|var\s+\w+\s*=)\b", text, re.I):
        return False
    if not is_code and (text.count("http") > 5 or text.count("www.") > 5):
        return False

    # Reject low-quality / off-topic content for a conversational model
    filters = corpus_cfg.get("filters", {})
    lower = text.lower()
    for kw in filters.get("reject_keywords", []):
        if kw.lower() in lower:
            return False
    for phrase in filters.get("reject_phrases", []):
        if phrase.lower() in lower:
            return False

    return True


def process_source_local_jsonl(
    name: str,
    source_cfg: Dict[str, Any],
    tokenizer,
    temp_dir: Path,
    corpus_cfg: Dict[str, Any],
    seq_len: int,
    max_samples: Optional[int] = None,
    target_chunks: Optional[int] = None,
) -> Tuple[int, List[Path]]:
    """Process a local JSONL file where each line has {text_col: text}."""
    jsonl_path = Path(source_cfg["local_jsonl"]).expanduser().resolve()
    text_col = source_cfg.get("text_col", "text")
    is_code = name == "code_stack"
    is_english = name == "fineweb_en_sample"
    buffer_chunks = corpus_cfg.get("buffer_chunks", 10000)

    print(f"\n[LOAD] Local JSONL source: {name} ({jsonl_path})")
    if not jsonl_path.exists():
        print(f"  [ERROR] File not found: {jsonl_path}")
        return 0, []

    current_tokens: List[int] = []
    all_chunk_files: List[Path] = []
    doc_count = 0
    filtered_count = 0
    chunk_count = 0
    chunk_file_idx = 0
    current_file = temp_dir / f"{name}_chunks_{chunk_file_idx}.npy"
    file_chunks: List[List[int]] = []

    start_time = time.time()
    last_log_time = start_time

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and doc_count >= max_samples:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = item.get(text_col, "")
            if isinstance(text, list):
                # For conversation-style records, concatenate all turns into a single text
                text = " ".join(str(t) for t in text if isinstance(t, str))

            doc_count += 1

            if not text or not isinstance(text, str):
                continue

            if not filter_document(text, corpus_cfg, is_code=is_code, is_english=is_english):
                continue

            filtered_count += 1
            tokens = tokenizer.encode(text, add_special_tokens=False)
            current_tokens.extend(tokens)

            while len(current_tokens) >= seq_len:
                chunk = current_tokens[:seq_len]
                file_chunks.append(chunk)
                current_tokens = current_tokens[seq_len:]
                chunk_count += 1

                if target_chunks and chunk_count >= target_chunks:
                    if file_chunks:
                        np.save(current_file, np.array(file_chunks, dtype=np.int32))
                        all_chunk_files.append(current_file)
                    print(f"  [DONE] {name}: target_chunks={target_chunks} reached")
                    return chunk_count, all_chunk_files

                if len(file_chunks) >= buffer_chunks:
                    np.save(current_file, np.array(file_chunks, dtype=np.int32))
                    all_chunk_files.append(current_file)
                    print(f"  [FLUSH] {name}: {chunk_count} chunks -> {current_file}")
                    file_chunks = []
                    chunk_file_idx += 1
                    current_file = temp_dir / f"{name}_chunks_{chunk_file_idx}.npy"
                    gc.collect()

            now = time.time()
            if now - last_log_time > 60:
                elapsed_min = (now - start_time) / 60
                print(
                    f"  [LOAD] {name}: {doc_count} docs, {filtered_count} passed, "
                    f"{chunk_count} chunks ({elapsed_min:.1f} min)"
                )
                last_log_time = now

    if file_chunks:
        np.save(current_file, np.array(file_chunks, dtype=np.int32))
        all_chunk_files.append(current_file)

    if current_tokens:
        print(f"  [INFO] {name}: {len(current_tokens)} leftover tokens discarded")

    elapsed = (time.time() - start_time) / 60
    print(
        f"  [DONE] {name}: {doc_count} docs, {filtered_count} passed, "
        f"{chunk_count} chunks, {len(all_chunk_files)} files ({elapsed:.1f} min)"
    )
    return chunk_count, all_chunk_files


def process_source_streaming(
    name: str,
    source_cfg: Dict[str, Any],
    tokenizer,
    temp_dir: Path,
    corpus_cfg: Dict[str, Any],
    seq_len: int,
    max_samples: Optional[int] = None,
    target_chunks: Optional[int] = None,
) -> Tuple[int, List[Path]]:
    hf_id = source_cfg["hf_id"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")
    streaming = source_cfg.get("streaming", True)
    text_col = source_cfg.get("text_col", "text")
    is_code = name == "code_stack"
    is_english = name == "fineweb_en_sample"
    source_max_samples = max_samples if max_samples is not None else source_cfg.get("max_samples")
    buffer_chunks = corpus_cfg.get("buffer_chunks", 10000)

    target_str = f", target_chunks={target_chunks}" if target_chunks else ""
    print(f"\n[LOAD] Streaming source: {name}{target_str}")
    try:
        if subset:
            ds = load_dataset(hf_id, subset, split=split, streaming=streaming)
        else:
            ds = load_dataset(hf_id, split=split, streaming=streaming)
    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return 0, []

    current_tokens: List[int] = []
    all_chunk_files: List[Path] = []
    doc_count = 0
    filtered_count = 0
    chunk_count = 0
    chunk_file_idx = 0
    current_file = temp_dir / f"{name}_chunks_{chunk_file_idx}.npy"
    file_chunks: List[List[int]] = []

    start_time = time.time()
    last_log_time = start_time

    for item in ds:
        if source_max_samples and doc_count >= source_max_samples:
            break

        text = item.get(text_col, "")
        doc_count += 1

        if not text or not isinstance(text, str):
            continue

        if not filter_document(text, corpus_cfg, is_code=is_code, is_english=is_english):
            continue

        filtered_count += 1
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)

        while len(current_tokens) >= seq_len:
            chunk = current_tokens[:seq_len]
            file_chunks.append(chunk)
            current_tokens = current_tokens[seq_len:]
            chunk_count += 1

            if target_chunks and chunk_count >= target_chunks:
                if file_chunks:
                    np.save(current_file, np.array(file_chunks, dtype=np.int32))
                    all_chunk_files.append(current_file)
                print(f"  [DONE] {name}: target_chunks={target_chunks} reached")
                return chunk_count, all_chunk_files

            if len(file_chunks) >= buffer_chunks:
                np.save(current_file, np.array(file_chunks, dtype=np.int32))
                all_chunk_files.append(current_file)
                print(f"  [FLUSH] {name}: {chunk_count} chunks -> {current_file}")
                file_chunks = []
                chunk_file_idx += 1
                current_file = temp_dir / f"{name}_chunks_{chunk_file_idx}.npy"
                gc.collect()

        now = time.time()
        if now - last_log_time > 60:
            elapsed_min = (now - start_time) / 60
            print(
                f"  [LOAD] {name}: {doc_count} docs, {filtered_count} passed, "
                f"{chunk_count} chunks ({elapsed_min:.1f} min)"
            )
            last_log_time = now

    if file_chunks:
        np.save(current_file, np.array(file_chunks, dtype=np.int32))
        all_chunk_files.append(current_file)

    if current_tokens:
        print(f"  [INFO] {name}: {len(current_tokens)} leftover tokens discarded")

    elapsed = (time.time() - start_time) / 60
    print(
        f"  [DONE] {name}: {doc_count} docs, {filtered_count} passed, "
        f"{chunk_count} chunks, {len(all_chunk_files)} files ({elapsed:.1f} min)"
    )
    return chunk_count, all_chunk_files


def mix_sources(
    temp_dir: Path,
    final_dir: Path,
    output_name: str,
    mix_ratios: Dict[str, float],
    seq_len: int,
    total_target_chunks: Optional[int] = None,
) -> Tuple[Path, int]:
    print("\n[MIX] Loading chunk counts from disk...")

    source_files: Dict[str, List[Path]] = {}
    for name in mix_ratios.keys():
        files = sorted(temp_dir.glob(f"{name}_chunks_*.npy"))
        if files:
            source_files[name] = files
            total = 0
            for f in files:
                arr = np.load(f, mmap_mode="r")
                total += arr.shape[0]
                del arr
            print(f"  [MIX] {name}: {total} chunks from {len(files)} files")

    total_ratio = sum(mix_ratios.values())
    total_mixed: List[np.ndarray] = []

    for name, ratio in mix_ratios.items():
        if name not in source_files:
            print(f"  [WARN] {name}: no chunks, skipping")
            continue

        total_available = 0
        for f in source_files[name]:
            arr = np.load(f, mmap_mode="r")
            total_available += arr.shape[0]
            del arr

        if total_target_chunks and total_target_chunks > 0:
            target_count = int(total_target_chunks * (ratio / total_ratio))
        else:
            target_count = int(total_available * (ratio / total_ratio))
        target_count = min(target_count, total_available)
        pct = ratio / total_ratio * 100
        print(f"  [MIX] {name}: selecting {target_count}/{total_available} ({pct:.1f}%)")

        if target_count == 0:
            continue

        indices = np.random.choice(total_available, target_count, replace=False)
        indices.sort()

        idx_ptr = 0
        file_offset = 0
        for f in source_files[name]:
            arr = np.load(f, mmap_mode="r")
            file_len = arr.shape[0]

            end_ptr = idx_ptr
            while end_ptr < len(indices) and indices[end_ptr] < file_offset + file_len:
                end_ptr += 1

            if end_ptr > idx_ptr:
                local_indices = indices[idx_ptr:end_ptr] - file_offset
                selected = arr[local_indices]
                total_mixed.append(selected)
                idx_ptr = end_ptr

            file_offset += file_len
            del arr
            gc.collect()

    if not total_mixed:
        raise RuntimeError("No chunks selected; cannot create corpus.")

    print(f"\n[MIX] Concatenating {len(total_mixed)} arrays...")
    final_array = np.concatenate(total_mixed, axis=0)
    print(f"[MIX] Total chunks: {final_array.shape[0]}")

    print("[MIX] Shuffling...")
    np.random.shuffle(final_array)

    final_tensor = torch.from_numpy(final_array.copy())
    final_path = final_dir / f"{output_name}.pt"
    torch.save(final_tensor, final_path)

    total_tokens = int(final_array.shape[0]) * seq_len
    print(f"\n[SAVE] Final corpus: {final_path}")
    print(f"[SAVE] Chunks: {final_array.shape[0]}, Tokens: {total_tokens:,} (~{total_tokens/1e9:.2f}B)")

    return final_path, total_tokens


def build_corpus(
    tokenizer,
    corpus_cfg: Dict[str, Any],
    output_dir: Path,
    temp_dir: Path,
    seq_len: int,
    output_name: str = "turkish_corpus_v2",
    sample: bool = False,
    sample_size: int = 1000,
) -> Tuple[Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    sources_cfg = corpus_cfg["sources"]

    if sample:
        print(f"[SAMPLE MODE] Each source limited to {sample_size} documents.")
        max_samples = sample_size
    else:
        max_samples = None

    total_chunks = 0
    for name, source_cfg in sources_cfg.items():
        target_tokens = source_cfg.get("target_tokens")
        target_chunks = target_tokens // seq_len if target_tokens and not sample else None
        if "hf_id" in source_cfg:
            chunks, _ = process_source_streaming(
                name=name,
                source_cfg=source_cfg,
                tokenizer=tokenizer,
                temp_dir=temp_dir,
                corpus_cfg=corpus_cfg,
                seq_len=seq_len,
                max_samples=max_samples,
                target_chunks=target_chunks,
            )
        elif "local_jsonl" in source_cfg:
            chunks, _ = process_source_local_jsonl(
                name=name,
                source_cfg=source_cfg,
                tokenizer=tokenizer,
                temp_dir=temp_dir,
                corpus_cfg=corpus_cfg,
                seq_len=seq_len,
                max_samples=max_samples,
                target_chunks=target_chunks,
            )
        else:
            print(f"[WARN] Source {name} has neither hf_id nor local_jsonl; skipping.")
            chunks = 0
        total_chunks += chunks

    print(f"\n[INFO] Total chunks from all sources: {total_chunks}")

    total_target_chunks = (
        sum(source_cfg.get("target_tokens", 0) for source_cfg in sources_cfg.values()) // seq_len
    )

    final_path, total_tokens = mix_sources(
        temp_dir=temp_dir,
        final_dir=output_dir,
        output_name=output_name,
        mix_ratios=corpus_cfg["mix_ratios"],
        seq_len=seq_len,
        total_target_chunks=total_target_chunks,
    )

    meta = {
        "total_chunks": int(total_chunks),
        "final_tokens": int(total_tokens),
        "seq_len": int(seq_len),
        "vocab_size": len(tokenizer),
        "tokenizer": str(tokenizer.name_or_path),
        "mix_ratios": corpus_cfg["mix_ratios"],
        "sample_mode": sample,
    }
    meta_path = output_dir / f"{output_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Meta: {meta_path}")

    print("\n[CLEAN] Removing temp files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[CLEAN] Done: {temp_dir}")

    return final_path, meta
