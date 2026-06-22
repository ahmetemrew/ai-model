"""Train a clean Turkish SentencePiece tokenizer for my-50m-model."""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import sentencepiece as spm
from transformers import LlamaTokenizer

# Add project root so we can import corpus utilities for filtering.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corpus import ENGLISH_CHARS, TURKISH_CHARS  # noqa: E402


# Special token set aligned with Llama-3 chat template.
SPECIAL_TOKENS = [
    "<PAD>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def _is_broken(text: str) -> bool:
    return "\ufffd" in text or "�" in text


def _char_ratio(text: str, chars: set) -> float:
    total = len(text) or 1
    return sum(1 for c in text if c in chars) / total


def _basic_filter(text: str, is_english: bool = False) -> bool:
    if not text or not isinstance(text, str):
        return False
    if len(text) < 50 or len(text) > 12000:
        return False
    if _is_broken(text):
        return False
    if "\x00" in text:
        return False
    if text.count("http") > 5 or text.count("www.") > 5:
        return False
    if re.search(r"\b(lorem ipsum|javascript:|function\s*\(|var\s+\w+\s*=)\b", text, re.I):
        return False

    if is_english:
        if _char_ratio(text, ENGLISH_CHARS) < 0.25:
            return False
    else:
        if _char_ratio(text, TURKISH_CHARS) < 0.30:
            return False
    return True


def _stream_hf_source(name: str, source_cfg: Dict[str, Any]) -> Iterator[str]:
    from datasets import load_dataset

    hf_id = source_cfg["hf_id"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")
    streaming = source_cfg.get("streaming", True)
    text_col = source_cfg.get("text_col", "text")
    is_english = name == "fineweb_en_sample"

    try:
        if subset:
            ds = load_dataset(hf_id, subset, split=split, streaming=streaming)
        else:
            ds = load_dataset(hf_id, split=split, streaming=streaming)
    except Exception as e:
        print(f"  [WARN] Could not load {name}: {e}")
        return

    for item in ds:
        text = item.get(text_col, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if isinstance(t, str))
        if not isinstance(text, str):
            continue
        if _basic_filter(text, is_english=is_english):
            yield text


def _stream_local_jsonl(name: str, source_cfg: Dict[str, Any]) -> Iterator[str]:
    jsonl_path = Path(source_cfg["local_jsonl"]).expanduser().resolve()
    text_col = source_cfg.get("text_col", "text")
    if not jsonl_path.exists():
        print(f"  [WARN] Local file not found for {name}: {jsonl_path}")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = item.get(text_col, "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text if isinstance(t, str))
            if isinstance(text, str) and _basic_filter(text, is_english=False):
                yield text


def stream_training_texts(
    corpus_cfg: Dict[str, Any],
    max_chars: int = 100_000_000,
    source_cap_chars: Optional[int] = None,
) -> Iterator[str]:
    """Yield filtered text examples for tokenizer training up to max_chars total."""
    sources_cfg = corpus_cfg["sources"]
    total_chars = 0
    target_per_source = (
        source_cap_chars if source_cap_chars is not None else max_chars // max(1, len(sources_cfg))
    )

    for name, source_cfg in sources_cfg.items():
        if total_chars >= max_chars:
            break

        source_max_chars = min(target_per_source, max_chars - total_chars)
        source_chars = 0
        print(f"[TOKENIZER-TRAIN] Collecting from {name} (target ~{source_max_chars:,} chars)...")

        if "hf_id" in source_cfg:
            gen = _stream_hf_source(name, source_cfg)
        elif "local_jsonl" in source_cfg:
            gen = _stream_local_jsonl(name, source_cfg)
        else:
            print(f"  [WARN] Unknown source type for {name}")
            continue

        for text in gen:
            if source_chars >= source_max_chars:
                break
            yield text
            source_chars += len(text)
            total_chars += len(text)
            if total_chars >= max_chars:
                break

    print(f"[TOKENIZER-TRAIN] Total collected: {total_chars:,} chars")


def build_sentencepiece_tokenizer(
    texts: Iterator[str],
    output_dir: Path,
    vocab_size: int = 50176,
    special_tokens: Optional[List[str]] = None,
) -> LlamaTokenizer:
    """Train a SentencePiece Unigram tokenizer and wrap it as LlamaTokenizer."""
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training texts to a UTF-8 temp file; SentencePiece reads files natively.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_file = tmpdir / "train.txt"
        prefix = tmpdir / "sp_tokenizer"

        print(f"[TOKENIZER-TRAIN] Writing training text to {input_file}...")
        with open(input_file, "w", encoding="utf-8", newline="\n") as f:
            for text in texts:
                # SentencePiece expects one sentence per line; normalize whitespace.
                line = " ".join(text.split())
                if not line:
                    continue
                f.write(line)
                f.write("\n")

        print("[TOKENIZER-TRAIN] Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=str(input_file),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            input_sentence_size=10_000_000,
            shuffle_input_sentence=True,
            user_defined_symbols=special_tokens,
            normalization_rule_name="identity",  # Do not alter Turkish characters.
            max_sentence_length=12000,
            byte_fallback=True,  # Represent rare chars/newlines as bytes for perfect round-trip.
            escape_whitespaces=True,  # Required by SentencePiece when byte_fallback is enabled.
            remove_extra_whitespaces=False,  # Preserve \n\n separators in chat template.
            add_dummy_prefix=False,  # Avoid leading whitespace artifacts.
            num_threads=os.cpu_count() or 1,
        )

        model_file = f"{prefix}.model"
        print(f"[TOKENIZER-TRAIN] Loading model: {model_file}")
        tokenizer = LlamaTokenizer(vocab_file=model_file, legacy=False)

    # Set special tokens to Llama-3 chat conventions.
    tokenizer.pad_token = "<PAD>"
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|eot_id|>"
    # Chat template explicitly emits the bos token; disable automatic bos insertion
    # so that encode/decode round-trips are stable with SentencePiece whitespace handling.
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    # Chat template (Llama-3 style). A single space after {{ bos_token }} is kept
    # so that the SentencePiece decoder reproduces the rendered string exactly.
    tokenizer.chat_template = (
        "{{ bos_token }} "
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|system|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'user' %}"
        "<|user|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% endif %}"
        "{% endfor %}"
    )

    tokenizer.save_pretrained(output_dir)
    print(f"[TOKENIZER-TRAIN] Saved tokenizer to: {output_dir}")
    return tokenizer


def validate_tokenizer(tokenizer: LlamaTokenizer) -> bool:
    """Run Gate-1 checks. Returns True if all pass."""
    print("\n[TOKENIZER-VALIDATE] Running Gate 1 checks...")

    # Reconfigure stdout so Windows terminals can print Turkish chars cleanly.
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    checks = []

    # 1. Single Turkish chars round-trip.
    chars = "abcçdefgğhıijklmnoöprsştuüvyzâîûABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZÂÎÛ"
    failed_chars = [
        c
        for c in chars
        if tokenizer.decode(tokenizer.encode(c, add_special_tokens=False, return_tensors=None)) != c
    ]
    ok = not failed_chars
    checks.append(("Turkish char round-trip", ok, f"failed={failed_chars[:10]}"))
    print(f"  Turkish char round-trip: {'PASS' if ok else 'FAIL'} ({len(failed_chars)} failed)")
    if failed_chars:
        print(f"    failed chars: {failed_chars!r}")

    # 2. Turkish words round-trip.
    words = ["Türkçe", "İstanbul", "şüphe", "çoğul", "ağaç", "düşünmek", "özgürlük", "ğıdıklamak"]
    failed_words = [
        w
        for w in words
        if tokenizer.decode(tokenizer.encode(w, add_special_tokens=False, return_tensors=None)) != w
    ]
    ok = not failed_words
    checks.append(("Turkish word round-trip", ok, f"failed={failed_words[:5]}"))
    print(f"  Turkish word round-trip: {'PASS' if ok else 'FAIL'} ({len(failed_words)} failed)")

    # 3. Long Turkish text round-trip.
    long_text = (
        "Türkiye Cumhuriyeti, Doğu Avrupa ve Batı Asya'da yer alan, Türkçe'nin "
        "resmî dil olduğu üniter bir devlettir. Başkenti Ankara'dır; en büyük şehri "
        "İstanbul'dur. Ülke, kuzeybatıda Bulgaristan ve Yunanistan, doğuda Gürcistan, "
        "Ermenistan, Azerbaycan ve İran, güneydoğuda Irak ve Suriye ile komşudur. "
        "Güneyinde Akdeniz, batısında Ege Denizi, kuzeyinde Karadeniz bulunur."
    )
    encoded = tokenizer.encode(long_text, add_special_tokens=False, return_tensors=None)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    ok = decoded == long_text
    checks.append(("Long Turkish round-trip", ok, None))
    print(f"  Long Turkish round-trip: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected: {long_text[:200]!r}")
        print(f"    decoded:  {decoded[:200]!r}")

    # 4. English round-trip.
    en_text = "The quick brown fox jumps over the lazy dog."
    ok = (
        tokenizer.decode(tokenizer.encode(en_text, add_special_tokens=False, return_tensors=None))
        == en_text
    )
    checks.append(("English round-trip", ok, None))
    print(f"  English round-trip: {'PASS' if ok else 'FAIL'}")

    # 5. Chat template round-trip.
    messages = [
        {"role": "system", "content": "Sen yardımcı bir asistansın."},
        {"role": "user", "content": "Merhaba!"},
        {"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer.encode(rendered, add_special_tokens=False, return_tensors=None)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    ok = decoded == rendered
    checks.append(("Chat template round-trip", ok, None))
    print(f"  Chat template round-trip: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected: {rendered[:300]!r}")
        print(f"    decoded:  {decoded[:300]!r}")

    # 6. Vocab size and special token IDs.
    ok = len(tokenizer) == 50176
    checks.append(("Vocab size == 50176", ok, f"got={len(tokenizer)}"))
    print(f"  Vocab size: {'PASS' if ok else 'FAIL'} ({len(tokenizer)})")

    for name, token in [("pad", tokenizer.pad_token), ("bos", tokenizer.bos_token), ("eos", tokenizer.eos_token)]:
        if token is None:
            checks.append((f"{name}_token exists", False, "None"))
            print(f"  {name}_token exists: FAIL (None)")
        else:
            tid = tokenizer.convert_tokens_to_ids(token)
            checks.append((f"{name}_token id", True, f"{token}={tid}"))
            print(f"  {name}_token: {token} (id={tid})")

    all_pass = all(ok for _, ok, _ in checks)
    print(f"\n[TOKENIZER-VALIDATE] Gate 1: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass
