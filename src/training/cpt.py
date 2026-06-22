import glob
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, LlamaForCausalLM, get_cosine_schedule_with_warmup

from src.model.model_50m import build_model, count_parameters, load_model_config
from src.training.dataset import PreTokenizedDataset
from src.utils.config import load_yaml, resolve_path


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_step(d: str) -> int:
    return int(Path(d).name.split("_")[-1])


def load_checkpoint(
    checkpoint_dir: Path,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> int:
    """Load the latest checkpoint and return the step to resume from.

    Falls back to best_cpt if it is newer than the latest model_step_* checkpoint.
    """
    dirs = sorted(glob.glob(str(checkpoint_dir / "model_step_*")), key=_checkpoint_step)

    # Determine the latest model_step step (0 if none)
    latest_step = _checkpoint_step(dirs[-1]) if dirs else 0

    # Check best_cpt step from status file or a step marker inside best_cpt
    best_ckpt = checkpoint_dir / "best_cpt"
    status_path = checkpoint_dir.parent / "checkpoint_status.json"
    best_step_marker = best_ckpt / "checkpoint_step.txt"
    best_step = 0
    if best_step_marker.exists():
        try:
            best_step = int(best_step_marker.read_text(encoding="utf-8").strip())
        except Exception:
            best_step = 0
    elif best_ckpt.exists() and status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            best_step = int(status.get("cpt_last_step", 0))
        except Exception:
            best_step = 0

    # Use best_cpt if it is newer than the latest model_step checkpoint
    # or if no model_step checkpoints exist at all.
    if best_ckpt.exists() and (best_step > latest_step or latest_step == 0):
        print(f"[CPT] Resuming from best checkpoint step {best_step}: {best_ckpt}")
        tmp = LlamaForCausalLM.from_pretrained(best_ckpt)
        model.load_state_dict(tmp.state_dict())
        del tmp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Optimizer state is not available for best_cpt; start fresh.
        return best_step + 1

    if not dirs:
        return 0

    latest = dirs[-1]
    step = _checkpoint_step(latest)
    print(f"[CPT] Resuming from checkpoint step {step}: {latest}")

    tmp = LlamaForCausalLM.from_pretrained(latest)
    model.load_state_dict(tmp.state_dict())
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    opt_file = checkpoint_dir / f"opt_step_{step}.pt"
    if opt_file.exists():
        st = torch.load(opt_file, map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])

    return step + 1


def save_checkpoint(
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    checkpoint_dir: Path,
):
    save_dir = checkpoint_dir / f"model_step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_dir / f"opt_step_{step}.pt",
    )
    print(f"  [CKPT] Saved step {step} -> {save_dir}")


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    dirs = sorted(glob.glob(str(checkpoint_dir / "model_step_*")), key=_checkpoint_step)
    for old in dirs[:-keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)
        opt_old = checkpoint_dir / f"opt_step_{Path(old).name.split('_')[-1]}.pt"
        if opt_old.exists():
            os.remove(opt_old)


@torch.no_grad()
def evaluate(model: LlamaForCausalLM, loader: DataLoader, max_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        count += 1
        if count >= max_batches:
            break
    model.train()
    return total_loss / count if count > 0 else float("inf")


@torch.no_grad()
def live_test(
    model: LlamaForCausalLM,
    tokenizer,
    prompt: str,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    model.train()
    return generated


def train_cpt(
    config_path: str | Path,
    resume: bool = True,
    max_steps: int | None = None,
    micro_batch: int | None = None,
):
    cfg = load_yaml(config_path)
    model_cfg = load_yaml("configs/model_50m.yaml")
    cpt_cfg = cfg["cpt"]

    torch.manual_seed(cfg.get("seed", 42))

    device = get_device()
    print(f"[CPT] Device: {device}")

    # Tokenizer
    tokenizer_dir = resolve_path(cfg["tokenizer_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Model
    config = load_model_config(model_cfg, tokenizer)
    model = build_model(config)
    model.gradient_checkpointing_enable()
    model = model.to(device)

    counts = count_parameters(model)
    print(f"[CPT] Model parameters: {counts['total']:,} (~{counts['total']/1e6:.1f}M)")

    # Corpus
    corpus_path = resolve_path(cfg["pretokenized_corpus"])
    print(f"[CPT] Loading corpus: {corpus_path}")
    corpus_tensor = torch.load(corpus_path, map_location="cpu")
    print(f"[CPT] Corpus shape: {corpus_tensor.shape}")

    dataset = PreTokenizedDataset(corpus_tensor, cfg["seq_len"], tokenizer.pad_token_id)
    val_size = max(1, int(len(dataset) * cpt_cfg["val_ratio"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if micro_batch is None:
        if device == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            micro_batch = cpt_cfg["micro_batch"] if total_vram >= 20 else cpt_cfg["micro_batch_t4"]
        else:
            micro_batch = cpt_cfg["micro_batch_t4"]

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        shuffle=True,
        drop_last=True,
        num_workers=cpt_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch,
        shuffle=False,
        num_workers=cpt_cfg["num_workers"],
    )
    print(f"[CPT] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, batch size: {micro_batch}")

    # Optimizer & scheduler
    max_steps = max_steps if max_steps is not None else cpt_cfg["max_steps"]
    warmup_steps = cpt_cfg["warmup_steps"]
    lr = cpt_cfg["lr"]
    min_lr = cpt_cfg["min_lr"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(cpt_cfg["betas"]),
        weight_decay=cpt_cfg["weight_decay"],
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr / lr) + (1 - min_lr / lr) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda") if device == "cuda" else None

    # Checkpoint setup
    checkpoint_dir = resolve_path(cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    status_path = checkpoint_dir.parent / "checkpoint_status.json"

    start_step = 0
    if resume:
        start_step = load_checkpoint(checkpoint_dir, model, optimizer, scheduler)
        print(f"[CPT] Resuming from step {start_step}")
        # Fast-forward scheduler so that resuming from best_cpt (where scheduler
        # state is not available) does not restart the warmup schedule.
        if start_step > 0:
            scheduler.last_epoch = start_step - 1

    model.train()
    global_step = start_step
    running_loss = 0.0
    best_val_loss = float("inf")

    # Training loop
    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=(device == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / cpt_cfg["grad_accum"]

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * cpt_cfg["grad_accum"]

            if (global_step + 1) % cpt_cfg["grad_accum"] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cpt_cfg["max_grad_norm"])
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if (global_step + 1) % cpt_cfg["log_every"] == 0:
                    avg_loss = running_loss / cpt_cfg["log_every"]
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"Step {global_step+1}/{max_steps} | loss={avg_loss:.4f} | lr={lr_now:.2e}")
                    running_loss = 0.0

                if (global_step + 1) % cpt_cfg["save_every"] == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step + 1, checkpoint_dir)
                    cleanup_old_checkpoints(checkpoint_dir, cpt_cfg["keep_last_n_checkpoints"])
                    status = {"cpt_last_step": global_step + 1}
                    with open(status_path, "w", encoding="utf-8") as f:
                        json.dump(status, f, indent=2)

                if (global_step + 1) % cpt_cfg["eval_every"] == 0:
                    val_loss = evaluate(model, val_loader)
                    print(f"  [VAL] loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = checkpoint_dir / "best_cpt"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir, safe_serialization=True)
                        (best_dir / "checkpoint_step.txt").write_text(str(global_step + 1), encoding="utf-8")
                        print(f"  [BEST] New best checkpoint: {val_loss:.4f}")
                    generated = live_test(model, tokenizer, cpt_cfg["live_test_prompt"])
                    print(f"  [LIVE] '{generated}'")

            global_step += 1
            if global_step >= max_steps:
                break

        print(f"[CPT] Epoch ended at step {global_step}")

    save_checkpoint(model, optimizer, scheduler, max_steps, checkpoint_dir)
    print("[CPT] Training complete.")