from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def load_model_and_tokenizer(model_dir: str | Path, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_chat(
    model: LlamaForCausalLM,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 48,
    temperature: float = 0.6,
    top_p: float = 0.85,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(out[0], skip_special_tokens=False)

    # Extract assistant response after the last assistant marker
    if "<|assistant|>" in generated:
        response = generated.split("<|assistant|>")[-1]
        for token in ["<|eot_id|>", "<|end_of_text|>", "<|user|>", "<|system|>"]:
            response = response.split(token)[0]
        response = response.strip()
    else:
        response = generated.replace(prompt, "").strip()

    return response
