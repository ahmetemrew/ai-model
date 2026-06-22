#!/usr/bin/env python3
"""Run supervised fine-tuning (SFT) for my-50m-model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import login

from src.training.sft import train_sft
from src.utils.config import get_hf_token


def main():
    parser = argparse.ArgumentParser(description="Run SFT for my-50m-model")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    parser.add_argument("--cpt-checkpoint", default=None, help="Path to CPT checkpoint (default: best in output_dir)")
    parser.add_argument("--no-resume", action="store_true", help="Start SFT from scratch instead of resuming from latest epoch")
    parser.add_argument("--epochs", type=int, default=None, help="Override SFT epochs (for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override SFT batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit SFT samples (for smoke tests)")
    args = parser.parse_args()

    token = get_hf_token()
    if token:
        print("[HF] Logging in with provided token.")
        login(token=token)

    train_sft(
        args.config,
        cpt_checkpoint=args.cpt_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
