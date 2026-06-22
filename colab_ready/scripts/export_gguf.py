#!/usr/bin/env python3
"""Export a trained HuggingFace model to GGUF format.

Produces multiple quantization files (FP16, Q8_0, Q4_K_M, etc.) directly on
Google Drive (or local disk) so they can be used with llama.cpp, Ollama,
LM Studio, etc.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a shell command and print it."""
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Export HF model to GGUF")
    parser.add_argument("--model-dir", default="models/trchat-50m-final", help="Path to trained HF model")
    parser.add_argument("--out-dir", default="models/gguf", help="Output directory for GGUF files")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp", help="Path to llama.cpp checkout")
    parser.add_argument(
        "--quantizations",
        default="fp16,q8_0,q4_k_m,q5_k_m",
        help="Comma-separated quantization types (e.g. fp16,q8_0,q4_k_m,q5_k_m,q4_0)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp = Path(args.llama_cpp_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Clone llama.cpp if it is not already present
    if not llama_cpp.exists():
        print("[INFO] Cloning llama.cpp...")
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_cpp),
        ])

    # Install conversion dependencies
    print("[INFO] Installing conversion dependencies...")
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "sentencepiece", "protobuf", "numpy", "gguf",
    ])

    # Convert to FP16 GGUF
    fp16_path = out_dir / f"{model_dir.name}-fp16.gguf"
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError("llama.cpp conversion script not found")

    print("[INFO] Converting to FP16 GGUF...")
    run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(fp16_path),
        "--outtype", "f16",
    ])

    # Build llama-quantize binary
    quantize_bin = llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        print("[INFO] Building llama-quantize...")
        run(["make", "-j", "llama-quantize"], cwd=llama_cpp)

    # Generate requested quantized files
    requested = [q.strip().lower() for q in args.quantizations.split(",")]
    for q in requested:
        if q == "fp16":
            continue
        out_file = out_dir / f"{model_dir.name}-{q}.gguf"
        print(f"[INFO] Quantizing {q}...")
        run([str(quantize_bin), str(fp16_path), str(out_file), q])

    print(f"\n[DONE] GGUF files saved to: {out_dir}")
    for f in sorted(out_dir.glob("*.gguf")):
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
