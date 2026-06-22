import glob
import math
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

from src.model.model_50m import build_model, load_model_config
from src.training.dataset import SFTChatDataset
from src.utils.config import load_yaml, resolve_path


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sft_checkpoint(checkpoint_dir: Path, model: LlamaForCausalLM) -> int:
    """Load the latest SFT epoch checkpoint and return the next epoch index (0-based)."""
    sft_dirs = sorted(
        glob.glob(str(checkpoint_dir / "sft_epoch_*")),
        key=lambda d: int(Path(d).name.split("_")[-1]),
    )
    if not sft_dirs:
        return 0
    latest = sft_dirs[-1]
    epoch = int(Path(latest).name.split("_")[-1])
    print(f"[SFT] Resuming from epoch {epoch}: {latest}")
    tmp = LlamaForCausalLM.from_pretrained(latest)
    model.load_state_dict(tmp.state_dict())
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return epoch


def train_sft(
    config_path: str | Path,
    cpt_checkpoint: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    resume: bool = True,
):
    cfg = load_yaml(config_path)
    model_cfg = load_yaml("configs/model_50m.yaml")
    sft_cfg = cfg["sft"]

    torch.manual_seed(cfg.get("seed", 42))

    device = get_device()
    print(f"[SFT] Device: {device}")

    # Tokenizer
    tokenizer_dir = resolve_path(cfg["tokenizer_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Model
    config = load_model_config(model_cfg, tokenizer)
    model = build_model(config)
    model = model.to(device)

    # Load CPT checkpoint if available
    checkpoint_dir = resolve_path(cfg["output_dir"])
    if cpt_checkpoint is None:
        best_ckpt = checkpoint_dir / "best_cpt"
        ckpt_dirs = sorted(glob.glob(str(checkpoint_dir / "model_step_*")))
        if best_ckpt.exists():
            cpt_checkpoint = best_ckpt
        elif ckpt_dirs:
            cpt_checkpoint = ckpt_dirs[-1]

    if cpt_checkpoint:
        print(f"[SFT] Loading CPT checkpoint: {cpt_checkpoint}")
        tmp = LlamaForCausalLM.from_pretrained(cpt_checkpoint)
        model.load_state_dict(tmp.state_dict())
        del tmp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[SFT WARNING] No CPT checkpoint found, training from scratch.")

    # Resume from latest SFT epoch if requested
    start_epoch = 0
    if resume:
        start_epoch = load_sft_checkpoint(checkpoint_dir, model)

    # SFT data
    sft_path = resolve_path(cfg["sft_data"])
    print(f"[SFT] Loading data: {sft_path}")
    sft_ds = SFTChatDataset(sft_path, tokenizer, sft_cfg["max_len"], max_samples=max_samples)
    print(f"[SFT] Samples: {len(sft_ds)}")

    batch_size = batch_size if batch_size is not None else sft_cfg["batch_size"]
    if device == "cpu":
        batch_size = 2  # tiny for local smoke test

    sft_loader = DataLoader(
        sft_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=sft_cfg["num_workers"],
    )
    print(f"[SFT] Batches per epoch: {len(sft_loader)}, batch size: {batch_size}")

    # Optimizer & scheduler
    total_steps = len(sft_loader) * sft_cfg["epochs"]
    warmup_steps = sft_cfg["warmup_steps"]
    lr = sft_cfg["lr"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(sft_cfg["betas"]),
        weight_decay=sft_cfg["weight_decay"],
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda") if device == "cuda" else None

    model.train()
    global_step = 0
    epochs = epochs if epochs is not None else sft_cfg["epochs"]
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for batch in sft_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=(device == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(sft_loader)
        print(f"[SFT] Epoch {epoch+1}/{sft_cfg['epochs']} | Avg loss={avg_loss:.4f}")

        epoch_dir = checkpoint_dir / f"sft_epoch_{epoch+1}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        print(f"[SFT] Saved epoch checkpoint: {epoch_dir}")

    # Final model
    model_dir = resolve_path(cfg.get("model_dir", "./models/my-model-final"))
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    print(f"[SFT] Final model saved: {model_dir}")
