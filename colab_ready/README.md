# Colab'a Hazır Paket

Bu klasör, **Google Colab üzerinde çalıştırarak** kendi modelinizi eğitmek için gereken tüm kodları içerir.

> ⚠️ **Önemli:** Bu pakette eğitim verisi ve tokenizer **yer almaz**. Aşağıdaki adımları takip ederek kendi verinizi ve tokenizer'ınızı hazırlayıp Drive'a yüklemelisiniz.

## İçindekiler

- `configs/` — Eğitim ve model ayarları
- `data/` — CPT corpus ve SFT verisinin yükleneceği boş klasörler
- `models/` — Tokenizer ve nihai modelin kaydedileceği boş klasör
- `notebooks/Train_AI_Model_Colab.ipynb` — Hazır Colab notebooku
- `scripts/` — Eğitim scriptleri
- `src/` — Kaynak kodlar
- `requirements.txt` — Python bağımlılıkları

## Adım 1: Veri ve Tokenizer Hazırla

Bu paketi Colab'a yüklemeden önce aşağıdaki dosyaları hazırlamanız gerekir:

### 1.1 Tokenizer Eğit

Kendi diliniz için bir tokenizer eğitin:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/tokenizer_train.txt \
  --output-dir models/my-tokenizer \
  --vocab-size 50000 \
  --max-chars 200000000
```

Detaylı bilgi için ana şablonun `docs/02_tokenizer.md` dosyasına bakın.

### 1.2 CPT Corpus Oluştur

Ham metinlerinizi tokenize edin:

```bash
python scripts/build_corpus.py --config configs/training.yaml --output-name my_corpus
```

Bu işlem şu dosyaları üretir:

- `data/corpus/my_corpus.pt`
- `data/corpus/my_corpus_meta.json`

Detaylı bilgi için `docs/03_corpus.md` dosyasına bakın.

### 1.3 SFT Verisi Hazırla

Sohbet formatında JSONL dosyanızı `data/sft/sft_data.jsonl` yoluna koyun:

```json
{"messages": [
  {"role": "user", "content": "Merhaba"},
  {"role": "assistant", "content": "Merhaba! Nasıl yardımcı olabilirim?"}
]}
```

## Adım 2: Google Drive'a Yükle

Hazırlanan dosyalarla birlikte bu klasörün **tüm içeriğini** Google Drive kök dizinine `ai-model-colab` olarak yükleyin.

Son yapı şöyle olmalı:

```
My Drive/
  └── ai-model-colab/
        ├── configs/
        ├── data/
        │    ├── corpus/my_corpus.pt
        │    ├── corpus/my_corpus_meta.json
        │    └── sft/sft_data.jsonl
        ├── models/
        │    └── my-tokenizer/
        ├── notebooks/Train_AI_Model_Colab.ipynb
        ├── scripts/
        ├── src/
        ├── requirements.txt
        └── README.md
```

## Adım 3: Colab Notebookunu Çalıştır

1. [https://colab.research.google.com](https://colab.research.google.com) adresine git.
2. **Dosya → Not Defteri Aç** seç.
3. **Google Drive** sekmesinden şu yolu aç:
   ```
   My Drive / ai-model-colab / notebooks / Train_AI_Model_Colab.ipynb
   ```
4. GPU çalışma zamanı seçin (**T4** veya **L4**).
5. **Çalışma zamanı → Tümünü çalıştır** seçeneğini kullanabilirsiniz.

Notebook otomatik olarak:
1. Drive'ı bağlar.
2. Gereksinimleri kurar.
3. Veri dosyalarının varlığını kontrol eder (eksikse talimat gösterir).
4. Eğer `checkpoints/best_cpt` yoksa CPT yapar.
5. Eğer `models/my-model-final` yoksa SFT yapar.
6. Eğitilmiş modelle kısa bir test yapar.

## Eğitimden Sonra

Nihai model şu klasöre kaydedilecek:

```
My Drive / ai-model-colab / models / my-model-final /
```

Bu modeli isterseniz HuggingFace Hub'a yükleyebilir veya başka yerlerde kullanabilirsiniz.

## Kısayollar

- CPT yapmadan sadece SFT yapmak için notebook içindeki `RUN_CPT = False` olarak ayarlayın.
- SFT yapmadan sadece test yapmak için `RUN_SFT = False` olarak ayarlayın.
