from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def load_model_config(model_cfg: Dict[str, Any], tokenizer) -> LlamaConfig:
    """Create a LlamaConfig from YAML config, filling token ids from tokenizer."""
    cfg = {
        k: v
        for k, v in model_cfg.items()
        if k not in ("name", "model_type") and v is not None
    }

    cfg["vocab_size"] = len(tokenizer)
    cfg["pad_token_id"] = tokenizer.pad_token_id
    cfg["bos_token_id"] = tokenizer.bos_token_id
    cfg["eos_token_id"] = tokenizer.eos_token_id

    # Only pass supported keys to LlamaConfig to avoid FutureWarnings.
    supported = set(LlamaConfig().__dict__.keys())
    cfg = {k: v for k, v in cfg.items() if k in supported}

    return LlamaConfig(**cfg)


def build_model(config: LlamaConfig) -> LlamaForCausalLM:
    """Build a LlamaForCausalLM model from config."""
    return LlamaForCausalLM(config)


def count_parameters(model: LlamaForCausalLM) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_model_for_inference(model_dir: str | Path, device: str = "cpu") -> LlamaForCausalLM:
    """Load a saved model for inference."""
    model = LlamaForCausalLM.from_pretrained(model_dir)
    return model.to(device)
