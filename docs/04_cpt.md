# 04 - CPT (Continued Pre-Training)

CPT, modelinizin corpus üzerinden genel dil bilgisi öğrendiği aşamadır.

## Hazırlık

Öncelikle:

- Tokenizer hazır olmalı.
- Corpus `.pt` dosyası oluşmuş olmalı.
- `configs/training.yaml` içinde `pretokenized_corpus` ve `corpus_meta` doğru gösterilmeli.

## CPT'yi Başlat

Yerel makinede (GPU varsa):

```bash
python scripts/train_cpt.py --config configs/training.yaml
```

Google Colab'da:

```bash
!python scripts/train_cpt.py --config configs/training.yaml
```

## Hafıza sorunu yaşarsanız

GPU hafızası yetmezse batch boyutunu düşürün:

```bash
python scripts/train_cpt.py --config configs/training.yaml --micro-batch 24
```

T4 için otomatik 24, L4 için otomatik 48 seçilir. İhtiyaca göre ayarlayın.

## Checkpoint ve Resume

- Her 500 adımda `checkpoints/model_step_XXXX` kaydedilir.
- Her 1000 adımda validation yapılır ve en iyi model `checkpoints/best_cpt` olarak saklanır.
- Eğitim koptuğunda aynı komutu tekrar çalıştırın; en son checkpoint’ten devam eder.

## Ne zaman durmalı?

- `max_steps` (varsayılan 80.000) dolunca otomatik durur.
- Dilerseniz `--max-steps 40000` ile sınırlandırabilirsiniz.
- Validation loss artık düşmüyorsa veya krediniz azalıyorsa erken durup SFT’ye geçebilirsiniz.

## Beklenen Çıktı

```
[CPT] Resuming from step 0
Step 1000/80000 | loss=... | lr=...
[VAL] loss=...
[BEST] New best checkpoint: ...
```

Loss düzenli düşüyorsa her şey yolunda demektir.

## Sonraki Adım

[05_sft.md](05_sft.md) dosyasına geçin.
