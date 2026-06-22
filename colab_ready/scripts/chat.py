#!/usr/bin/env python3
"""Simple chat script to test the trained model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Chat with your trained model")
    parser.add_argument("--model-dir", default="models/my-model-final", help="Path to trained model directory")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CHAT] Loading model from {args.model_dir} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = LlamaForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    print("[CHAT] Ready. Type 'quit' or 'exit' to stop.\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response (after the last prompt)
        response = generated_text[len(prompt):].strip()
        print(f"Model: {response}\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
