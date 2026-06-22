#!/usr/bin/env python3
"""Download and prepare the SFT dataset in standard chat format."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import login

from src.data.sft import prepare_sft_data
from src.utils.config import get_hf_token, load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Prepare my-50m-model SFT data")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_path = resolve_path(cfg["sft_data"])
    data_dir = output_path.parent

    token = get_hf_token()
    if token:
        print("[HF] Logging in with provided token.")
        login(token=token)
    else:
        print("[HF] No token found. Public datasets only.")

    prepare_sft_data(output_path=output_path, data_dir=data_dir)
    print(f"\n[Done] SFT data ready at: {output_path}")


if __name__ == "__main__":
    main()
