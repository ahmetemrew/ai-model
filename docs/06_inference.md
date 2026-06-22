# 06 - Eğitilmiş Modeli Kullanma

SFT bittikten sonra modeliniz `models/my-model-final/` (veya config’teki `model_dir`) altında hazır olacak.

## Basit Test

```python
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_dir = "models/my-model-final"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

messages = [
    {"role": "user", "content": "Türkiye'nin başkenti neresidir?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Chat Formatı

Model, eğitim sırasında tokenizer’a eklenen `chat_template` ile uyumlu çalışır. Mutlaka `apply_chat_template` kullanın.

## HuggingFace Hub’a Yükleme

Modelinizi paylaşmak isterseniz:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="kullaniciadi/model-adi", exist_ok=True)
api.upload_folder(
    folder_path="models/my-model-final",
    repo_id="kullaniciadi/model-adi",
)
```

## Sonraki Adım

[07_colab.md](07_colab.md) dosyasına geçin.
