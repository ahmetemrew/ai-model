import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


BEHAVIOR_SAMPLES: List[Dict[str, Any]] = [
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Sence Türkiye'nin en güzel şehri hangisidir?"},
            {"role": "assistant", "content": "Bu tamamen kişisel tercihe bağlı, ama İstanbul, Kapadokya ve Pamukkale çok popüler."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Sana zarar verebilir miyim?"},
            {"role": "assistant", "content": "Hayır, zarar veremezsiniz. Ben bir yapay zeka asistanıyım."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Kuantum mekaniğini anlat."},
            {"role": "assistant", "content": "Kuantum mekaniği, atom ve altı parçacıkların davranışlarını açıklayan fizik dalıdır."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Bana şiir yaz."},
            {"role": "assistant", "content": "Gün batarken denizde,\nYalnız martı sesleri,\nHüzün dolu bir akşam."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Bugün hava nasıl?"},
            {"role": "assistant", "content": "Konum bilginiz olmadan hava durumunu söyleyemem."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Python'da liste nasıl oluşturulur?"},
            {"role": "assistant", "content": 'Python\'da liste köşeli parantezlerle oluşturulur: meyveler = ["elma", "armut", "muz"]'},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Türkiye'nin başkenti neresidir?"},
            {"role": "assistant", "content": "Türkiye'nin başkenti Ankara'dır."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "2+2 kaçtır?"},
            {"role": "assistant", "content": "2 + 2 = 4'tür."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Kendini tanıt."},
            {"role": "assistant", "content": "Ben yardımcı bir yapay zeka asistanıyım."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            {"role": "user", "content": "Yarın ne yapmalıyım?"},
            {"role": "assistant", "content": "Bu tamamen sizin hedeflerinize bağlı."},
        ]
    },
]


def normalize_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Convert various conversation formats to standard {'role': ..., 'content': ...}."""
    msgs = row.get("messages") or row.get("conversations") or []
    if len(msgs) >= 2:
        clean = []
        for m in msgs:
            role = m.get("role") or m.get("from") or "user"
            content = m.get("content") or m.get("value") or m.get("text") or ""
            if role in ("from", "human"):
                role = "user"
            if role in ("gpt", "bot"):
                role = "assistant"
            if role not in ("system", "user", "assistant"):
                continue
            clean.append({"role": role, "content": content})
        return clean if len(clean) >= 2 else None

    # Fallback: input/output pairs (InstrucTurca, turkish-dialogues, etc.)
    content_in = row.get("input") or row.get("Input") or ""
    content_out = row.get("output") or row.get("Output") or ""
    if content_in and content_out:
        return [
            {"role": "user", "content": content_in},
            {"role": "assistant", "content": content_out},
        ]

    return None


def download_sft_dataset(
    name: str,
    subset: Optional[str],
    split: str,
    output_file: Path,
    max_samples: int = 50_000,
) -> int:
    if output_file.exists():
        print(f"[SFT] {output_file.name} already exists, skipping download.")
        count = 0
        with open(output_file, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    print(f"[SFT] Downloading {name} (subset={subset})...")
    try:
        if subset:
            ds = load_dataset(name, subset, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
    except Exception as e:
        print(f"[SFT ERROR] {name}: {e}")
        return 0

    count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            msgs = normalize_messages(row)
            if msgs is None:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            count += 1

    print(f"[SFT] {name}: {count} samples -> {output_file}")
    return count


def write_behavior_data(output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in BEHAVIOR_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[SFT] Behavior samples: {len(BEHAVIOR_SAMPLES)} -> {output_file}")
    return len(BEHAVIOR_SAMPLES)


def prepare_sft_data(
    output_path: Path,
    data_dir: Path,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = [
            {"name": "turkish-nlp-suite/InstrucTurca", "subset": None, "split": "train", "file": "sft_instruc.jsonl", "max_samples": 50_000},
            {"name": "bugrabilge/Bilge-Turkish-CoT-50K", "subset": None, "split": "train", "file": "sft_cot.jsonl", "max_samples": 15_000},
            {"name": "odmow/turkish-dialogues", "subset": None, "split": "train", "file": "sft_dialog.jsonl", "max_samples": 5_000},
        ]

    total = 0
    for src in sources:
        total += download_sft_dataset(
            name=src["name"],
            subset=src.get("subset"),
            split=src.get("split", "train"),
            output_file=data_dir / src["file"],
            max_samples=src.get("max_samples", 50_000),
        )

    behavior_file = data_dir / "sft_behavior.jsonl"
    total += write_behavior_data(behavior_file)

    sft_files = [data_dir / src["file"] for src in sources] + [behavior_file]
    with open(output_path, "w", encoding="utf-8") as fw:
        for fpath in sft_files:
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as fr:
                for line in fr:
                    fw.write(line)

    final_count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for _ in f:
            final_count += 1

    print(f"\n[SFT] Combined dataset: {final_count} samples -> {output_path}")
    return final_count
