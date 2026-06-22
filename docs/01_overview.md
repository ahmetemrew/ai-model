# 01 - Genel Bakış

Bu proje, kendi küçük dil modelinizi (LLM) sıfırdan eğitmeniz için bir şablondur.

## Hangi aşamalardan oluşur?

1. **Tokenizer Eğitimi** - Modelin anlayacağı alt kelime birimlerini oluşturur.
2. **Veri Seti Hazırlığı** - Ham metinleri ve sohbet verilerini toplar.
3. **Corpus Oluşturma** - Ham metinleri tokenize eder ve eğitim için hazır hale getirir.
4. **CPT (Continued Pre-Training)** - Modeli büyük corpus üzerinde genel dil bilgisi kazandırır.
5. **SFT (Supervised Fine-Tuning)** - Modeli sohbet formatına ve görevlere göre ince ayarlar.
6. **Test ve Kullanım** - Eğitilmiş modeli kullanarak çıktı üretir.

## Kimler için?

- Kendi dilinde küçük bir model eğitmek isteyenler
- LLM eğitim sürecini öğrenmek isteyenler
- Veri seti ve tokenizer üzerinde deney yapmak isteyenler

## Zaman ve Maliyet Tahmini

| Aşama | Yaklaşık Süre (Colab T4) | Maliyet |
|-------|--------------------------|---------|
| Tokenizer | 30–60 dk | Ücretsiz (CPU) |
| Corpus build | 2–6 saat | Ücretsiz (CPU) |
| CPT (80k adım) | 25–35 saat | ~20–40 kredi |
| SFT (5 epoch) | 3–6 saat | ~5–15 kredi |

L4 kullanırsanız süreler yaklaşık yarıya iner ama kredi tüketimi artar.

## Sonraki Adım

[02_tokenizer.md](02_tokenizer.md) dosyasına geçin.
