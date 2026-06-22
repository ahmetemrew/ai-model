# 05 - SFT (Supervised Fine-Tuning)

SFT, CPT ile eğitilmiş modeli sohbet ve görev formatına sokan aşamadır.

## SFT Verisi Formatı

`data/sft/sft_data.jsonl` dosyası şu formatta olmalı:

```json
{"messages": [
  {"role": "user", "content": "Türkiye'nin başkenti neresidir?"},
  {"role": "assistant", "content": "Türkiye'nin başkenti Ankara'dır."}
]}
{"messages": [
  {"role": "user", "content": "Merhaba"},
  {"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"}
]}
```

Roller: `system`, `user`, `assistant`.

## Veriyi İndir veya Kendin Hazırla

Hazır bir dataset kullanmak için:

```bash
python scripts/prepare_sft_data.py --config configs/training.yaml
```

Kendi verinizi kullanmak için JSONL dosyanızı `data/sft/` altına koyun ve `configs/training.yaml` içinde `sft_data` yolunu güncelleyin.

## SFT'yi Başlat

```bash
python scripts/train_sft.py --config configs/training.yaml
```

Varsayılan olarak:
- CPT’nin `best_cpt` checkpoint’inden devam eder.
- 5 epoch eğitir.
- Her epoch sonunda `checkpoints/sft_epoch_X` kaydeder.
- Sonunda nihai modeli `models/my-model-final/` altına yazar.

## Daha Fazla Epoch

İlk 5 epoch yeterli gelmezse:

```bash
python scripts/train_sft.py --config configs/training.yaml --epochs 5
```

Bu komut 6. epoch’tan devam edip 10. epoch’a kadar gider.

## Sonraki Adım

[06_inference.md](06_inference.md) dosyasına geçin.
