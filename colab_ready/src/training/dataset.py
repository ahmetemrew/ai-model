import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """Dataset that loads a pretokenized corpus tensor and returns language-modeling samples."""

    def __init__(self, tensor: torch.Tensor, seq_len: int, pad_token_id: int):
        self.data = tensor
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.data[idx].long()
        return {
            "input_ids": ids.clone(),
            "labels": ids.clone(),
            "attention_mask": (ids != self.pad_token_id).long(),
        }


class SFTChatDataset(Dataset):
    """Dataset for supervised fine-tuning on chat-formatted JSONL data."""

    def __init__(self, jsonl_path: str | Path, tokenizer, max_len: int = 512, max_samples: int | None = None):
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.pad_token_id

        jsonl_path = Path(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                sample = self._build_sample(msgs)
                if sample is not None:
                    self.samples.append(sample)

    def _build_sample(self, msgs: List[Dict[str, str]]) -> Dict[str, torch.Tensor] | None:
        if len(msgs) < 2:
            return None

        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = ""
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                text += f"<|{role}|>\n\n{content}<|eot_id|>"

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [self.pad_token_id] * (self.max_len - len(ids))
        ids = torch.tensor(ids, dtype=torch.long)

        labels = ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": (ids != self.pad_token_id).long(),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
