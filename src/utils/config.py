import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    path = Path(path)
    if not path.is_absolute():
        path = get_project_root() / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_hf_token() -> str | None:
    """Read HuggingFace token from env or secrets.json."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    secret_paths = [
        get_project_root() / "secrets.json",
        Path.home() / ".config" / "huggingface" / "token",
    ]
    for p in secret_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data.get("hf_token") or data.get("token")
                return data.strip()
            except Exception:
                continue
    return None


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to project root if not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    root = root or get_project_root()
    return root / path


def merge_model_config(model_cfg: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Fill special token ids from tokenizer into model config."""
    cfg = dict(model_cfg)
    cfg["pad_token_id"] = tokenizer.pad_token_id
    cfg["bos_token_id"] = tokenizer.bos_token_id
    cfg["eos_token_id"] = tokenizer.eos_token_id
    cfg["vocab_size"] = len(tokenizer)
    return cfg
