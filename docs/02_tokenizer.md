# 02 - Tokenizer Eğitimi

Tokenizer, modelin metni anlayacağı sayılara çevirmesini sağlayan bileşendir. **Tokenizer kalitesi, model kalitesini doğrudan etkiler.**

## Nasıl çalışır?

Bu şablonda **SentencePiece Unigram** tokenizer kullanılır. SentencePiece:
- Doğrudan UTF-8 dosya okuduğu için Türkçe, Arapça, Çince gibi dillerde karakter bozulması yapmaz.
- `byte_fallback=True` sayesinde bilinmeyen karakterleri byte'lara böler.

## Kendi tokenizer'ını eğit

### 1. Ham metin dosyası hazırla

Tokenizer eğitimi için düz metin dosyası gerekir. Örnek:

```bash
mkdir -p data/raw
# data/raw/tokenizer_train.txt oluştur
# En az 50–100 milyon karakter önerilir.
```

### 2. Scripti çalıştır

```bash
python scripts/train_tokenizer.py \
  --input data/raw/tokenizer_train.txt \
  --output-dir models/my-tokenizer \
  --vocab-size 50000 \
  --max-chars 200000000
```

Parametreler:
- `--vocab-size`: Kelime haznesi boyutu (50.000 civarı iyi başlangıçtır).
- `--max-chars`: Eğitimde kullanılacak maksimum karakter sayısı.
- `--input`: Tokenizer eğitim metni.

### 3. Kalite kontrolü

Eğitim sonrası şu testleri yapın:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/my-tokenizer")

samples = [
    "Türkçe karakterler: çğıöşü",
    "İstanbul'un tarihi yerleri",
    "Bugün hava çok güzel.",
]

for s in samples:
    ids = tok.encode(s)
    dec = tok.decode(ids)
    print(s, "==>", dec, "| match:", s == dec)
```

Tüm örneklerde `match: True` çıkmalı.

### 4. Config'e ekle

`configs/training.yaml` dosyasında:

```yaml
tokenizer_dir: "./models/my-tokenizer"
```

## Önemli İpuçları

- **Kendi diliniz için tokenizer eğitin.** Başka bir dilin tokenizer'ını kullanmak kötü sonuç verir.
- Vocab boyutu 30.000–60.000 arası deneyebilirsiniz.
- Eğitim metni, hedef modelin göreceği metinlerle aynı dilden ve tarzdan olmalı.

## Sonraki Adım

[03_corpus.md](03_corpus.md) dosyasına geçin.
