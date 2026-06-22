# 07 - Google Colab ile Eğitim

Google Colab, ücretsiz veya ucuz GPU kiralayarak model eğitmenin en kolay yoludur.

## Hazırlık

1. Bu repo içindeki `colab_export/` klasörünü bilgisayarından Google Drive’ına şu şekilde yükle:

```
My Drive/
  └── ai-model-colab/
        ├── configs/
        ├── data/
        ├── models/
        ├── notebooks/
        ├── scripts/
        ├── src/
        ├── requirements.txt
        └── README.md
```

> Not: `data/corpus/*.pt` ve `data/sft/*.jsonl` gibi büyük dosyaları da yüklemelisin. Bunlar eğitim verisidir.

2. Google Colab’a git.
3. `notebooks/My_50M_Model_Colab.ipynb` dosyasını Drive’dan aç.

## Notebook Adımları

1. **Drive mount** hücresini çalıştır.
2. **Requirements** hücresini çalıştır.
3. **GPU seç:** `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 veya L4`.
4. **CPT hücresini** çalıştır.
5. CPT bitince **SFT hücresini** çalıştır.
6. Sonunda test hücresi ile modeli dene.

## Oturum Koparsa Ne Olur?

Colab oturumları 12 saatte bir kopabilir. Endişelenmeyin:

- Checkpointler her 500 adımda **Google Drive’a** kaydedilir.
- Aynı hücreyi tekrar çalıştırdığınızda en son checkpoint’ten devam eder.

## Kredi İpuçları

- T4 daha ekonomiktir.
- L4 daha hızlıdır.
- CPT uzun sürer; SFT daha kısa sürer.
- Krediniz azalıyorsa CPT’yi erken durdurup SFT’ye geçebilirsiniz.

## Sorun Giderme

### Out of Memory (OOM)

CPT hücresine `--micro-batch 24` ekleyin:

```bash
!python scripts/train_cpt.py --config configs/training.yaml --micro-batch 24
```

### Model yeniden baştan başlıyor

Eğer oturum koptuktan sonra step 0’dan başlıyorsa, `checkpoints/` klasörünün Drive’a kaydedildiğinden emin olun. `configs/training.yaml` içinde `output_dir: "./checkpoints"` olduğundan emin olun.

## Tebrikler

Artık kendi dil modelinizi eğitmek için gereken tüm adımları biliyorsunuz. İyi çalışmalar!
