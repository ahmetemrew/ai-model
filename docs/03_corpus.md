# 03 - Corpus Oluşturma

Corpus, modelin CPT aşamasında öğreneceği büyük metin koleksiyonudur. Bu şablonda corpus:

- Ham metin kaynaklarından (HuggingFace dataset veya yerel JSONL) okunur.
- Filtrelenir, temizlenir.
- Tokenize edilir.
- Belirlenen oranlarda karıştırılır.

## Veri Kaynakları

`configs/training.yaml` içinde `corpus.sources` altından kaynakları tanımlayın.

### HuggingFace dataset örneği

```yaml
my_dataset:
  hf_id: "username/dataset-name"
  subset: "tr"
  target_tokens: 1000000000
  text_col: "text"
  split: "train"
  streaming: true
```

### Yerel JSONL örneği

```yaml
my_local_data:
  local_jsonl: "./data/corpus/my_text.jsonl"
  target_tokens: 50000000
  text_col: "text"
```

JSONL formatı:

```json
{"text": "Bu bir örnek metindir."}
{"text": "İkinci örnek metin."}
```

## Filtreler

`corpus.filters` altında ayarlanır:

- `min_chars`: Minimum karakter sayısı.
- `max_chars`: Maksimum karakter sayısı.
- `turkish_char_ratio`: Hedef dil karakterlerinin minimum oranı.
- `english_char_ratio`: İngilizce karakterlerin maksimum oranı.
- `max_repetition_ratio`: Tekrar oranı üst sınırı.
- `reject_keywords`: Kabul edilmeyecek kelimeler listesi.
- `reject_phrases`: Kabul edilmeyecek ifadeler listesi.

Kendi dilinize ve verinize göre bu değerleri değiştirin.

## Mix Oranları

`corpus.mix_ratios` ile her kaynağın corpus içindeki ağırlığını ayarlarsınız. Toplamı 1.0 etmelidir.

```yaml
mix_ratios:
  my_dataset: 0.8
  my_local_data: 0.2
```

## Corpus'u Oluştur

```bash
python scripts/build_corpus.py --config configs/training.yaml --output-name my_corpus
```

Sonuçta şu dosyalar oluşur:

- `data/corpus/my_corpus.pt`
- `data/corpus/my_corpus_meta.json`

## Kalite Kontrolü

```bash
python scripts/analyze_final_corpus.py \
  --pt data/corpus/my_corpus.pt \
  --meta data/corpus/my_corpus_meta.json \
  --tokenizer models/my-tokenizer \
  --samples 20
```

Çıktıdaki metinleri okuyun. Bozuk karakter, garip tekrarlar veya spam içerik varsa filtreleri güçlendirin.

## Sonraki Adım

[04_cpt.md](04_cpt.md) dosyasına geçin.
