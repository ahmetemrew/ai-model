#!/usr/bin/env python3
"""Run continued pre-training (CPT) for my-50m-model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import login

from src.training.cpt import train_cpt
from src.utils.config import get_hf_token


def main():
    parser = argparse.ArgumentParser(description="Run CPT for my-50m-model")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming")
    parser.add_argument("--max-steps", type=int, default=None, help="Override CPT max steps (for smoke tests)")
    parser.add_argument("--micro-batch", type=int, default=None, help="Override micro batch size")
    args = parser.parse_args()

    token = get_hf_token()
    if token:
        print("[HF] Logging in with provided token.")
        login(token=token)

    train_cpt(
        args.config,
        resume=not args.no_resume,
        max_steps=args.max_steps,
        micro_batch=args.micro_batch,
    )


if __name__ == "__main__":
    main()
