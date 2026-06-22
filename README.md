# Kendi Dil Modelini Eğit - 50M Başlangıç Şablonu

Bu repo, **sıfırdan kendi küçük dil modelinizi (LLM) eğitmek** için adım adım bir şablondur.

Kendi veriniz, kendi tokenizer'ınız ve kendi chat formatınızla çalışabilirsiniz. Örnek olarak 50M parametreli bir model (Llama mimarisi) hedeflenmiştir, ancak aynı yapıyı başka diller ve farklı boyutlar için de kullanabilirsiniz.

## Özellikler

- SentencePiece tabanlı tokenizer eğitimi
- Büyük corpus oluşturma ve karıştırma
- CPT (Continued Pre-Training)
- SFT (Supervised Fine-Tuning / Chat eğitimi)
- Hazır Google Colab notebooku
- **Hazır Colab paketi** (`colab_ready/` alt klasörü)
- Otomatik checkpoint yedekleme (Google Drive üzerinden)

## Sistem Gereksinimleri

### Yalnızca dataset/tokenizer hazırlamak için

- Windows, macOS veya Linux
- Python 3.10+
- ~50 GB boş disk alanı (corpus için)

### Eğitim için

- **Önerilen:** Google Colab Pro / Pay-as-you-go (GPU: T4 veya L4)
- **Yerel:** CUDA destekli GPU (en az 16 GB VRAM)

> Not: Bu şablon eğitim için Colab üzerinden GPU kullanmayı varsayar. Dataset hazırlama yerel makinede yapılır.

## Hızlı Başlangıç

1. Repoyu klonlayın veya indirin.
2. Bağımlılıkları kurun:
   ```bash
   pip install -r requirements.txt
   ```
3. HuggingFace token ayarlayın (private/gated dataset kullanacaksanız):
   ```bash
   export HF_TOKEN=your_token_here
   ```
4. `configs/training.yaml` dosyasını kendi veri kaynaklarınıza göre düzenleyin.
5. [docs/](docs/) klasöründeki kılavuzları takip edin.

## Adım Adım Akış

| Adım | Açıklama | Klasör/Dosya |
|------|----------|--------------|
| 1 | Tokenizer eğit | `scripts/train_tokenizer.py` |
| 2 | Veri setini hazırla | `scripts/prepare_sft_data.py`, `src/data/corpus.py` |
| 3 | Corpus oluştur | `scripts/build_corpus.py` |
| 4 | CPT yap | `scripts/train_cpt.py` |
| 5 | SFT yap | `scripts/train_sft.py` |
| 6 | Test et | `notebooks/My_50M_Model_Colab.ipynb` veya kendi inference kodun |

## Proje Yapısı

```
.
├── colab_ready/           # Hazır Colab paketi (veri/tokenizer hariç)
│   ├── notebooks/
│   ├── scripts/
│   ├── src/
│   └── README.md
├── configs/
│   ├── training.yaml      # Ana eğitim ayarları
│   └── model_50m.yaml     # Model mimarisi
├── data/
│   ├── corpus/            # CPT corpus'u (.pt dosyası)
│   └── sft/               # SFT verisi (.jsonl)
├── docs/                  # Detaylı kılavuzlar
├── models/
│   └── my-tokenizer/      # Eğitilmiş tokenizer
├── notebooks/
│   └── My_50M_Model_Colab.ipynb  # Hazır Colab notebooku
├── scripts/               # Çalıştırılabilir scriptler
├── src/                   # Kaynak kod
├── requirements.txt
└── README.md
```

## Colab ile Tek Tıkla Başlat

Eğitim verinizi ve tokenizer'ınızı hazırladıktan sonra:

1. `colab_ready/` klasörünü Google Drive köküne `ai-model-colab` olarak yükleyin.
2. `colab_ready/notebooks/Train_AI_Model_Colab.ipynb` dosyasını Colab’te açın.
3. GPU çalışma zamanı seçin.
4. **Çalışma zamanı → Tümünü çalıştır** seçeneğini kullanın.

Detaylı talimatlar için `colab_ready/README.md` dosyasına bakın.

## Önemli Notlar

- `data/` ve `models/` klasörlerindeki büyük dosyalar `.gitignore` ile dışlanmıştır. Bunları GitHub'a yüklemeyin; eğitim sonrası modelleri HuggingFace Hub veya başka bir depolama servisine yükleyin.
- Google Colab kullanıyorsanız büyük dosyaları Google Drive'a yükleyin.
- Veri kaynaklarınızı ve hedef dilinizi `configs/training.yaml` içinden değiştirin.
- `colab_ready/` paketi **veri ve tokenizer içermez**. Kullanıcı kendi verisini hazırlar, talimatlar `colab_ready/README.md` içindedir.

## GitHub’a Yükleme

Bu repoyu GitHub’a yüklemek için:

```bash
cd ai-model
git init
git add .
git commit -m "Initial template commit"
git remote add origin https://github.com/ahmetemrew/ai-model.git
git branch -M main
git push -u origin main
```

> Not: `data/` ve `models/` ile `colab_ready/data/` ve `colab_ready/models/` içindeki büyük dosyalar `.gitignore` ile dışlanmıştır. Bunları GitHub’a yüklemeyin.

## Lisans

Kendi projenize göre uygun bir lisans seçin ve buraya ekleyin.

## Teşekkür

Bu şablon, küçük ölçekli dil modelleri eğitmek isteyenler için hazırlanmıştır. Kendi verinizle deneyin ve geliştirin.
