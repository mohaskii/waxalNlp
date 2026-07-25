# WAXAL ASR Challenge — Zindi Competition

End-to-end automatic speech recognition pipeline for **Lingala**, **Shona**, and **Luganda** using Wav2Vec2-MMS-1B, w2v-BERT 2.0 with LoRA fine-tuning, KenLM language model decoding, and logit-averaging ensemble.

**Target metric:** `0.5 × WER + 0.5 × CER`

---

## 📁 Project Structure

```
waxalNlp/
├── config.py                    # Central configuration (paths, hyperparameters)
├── download_data.py             # Step 0: Download audio from Hugging Face
├── step1_preprocessing.py       # Step 1: GroupKFold + normalization + vocab
├── step2_train_mms.py           # Step 2: Fine-tune MMS-1B with LoRA
├── step3_train_kenlm.py         # Step 3: Train KenLM 5-gram + tune alpha/beta
├── step3_decode_lm.py           # Step 3b: CTC beam search decode with pyctcdecode
├── step4_train_w2vbert.py       # Step 4: Fine-tune w2v-BERT 2.0 with LoRA
├── step5_ensemble.py            # Step 5: Logit averaging ensemble + submission
├── run_pipeline.py              # Master orchestrator for all steps
├── utils.py                     # Shared utilities (normalize, audio, metrics)
├── requirements.txt             # Python dependencies
├── pyrightconfig.json           # Type-checking config
├── .gitignore
└── README.md                    # ← You are here
```

---

## 🚀 Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `transformers`, `datasets`, `peft` — Hugging Face ecosystem
- `torch` — Deep learning framework
- `pyctcdecode`, `kenlm` — CTC beam search with language model
- `librosa`, `soundfile` — Audio loading
- `jiwer` — WER / CER computation
- `scikit-learn` — GroupKFold cross-validation

### 2. Data

The competition provides two CSV files and you fetch audio from Hugging Face:

```
data/
├── Train.csv            # Training metadata (id, transcription, language, original_split)
├── Test.csv             # Test IDs (ID column only)
├── SampleSubmission.csv # Submission template
├── Train/               # Audio files (downloaded by step 0)
│   ├── lug_96123.wav
│   ├── lin_4521.wav
│   └── ...
└── Test/                # Audio files (downloaded by step 0)
    ├── lug_96114.wav
    └── ...
```

Place `Train.csv` and `Test.csv` into `data/` (you can set a custom path via `WAXAL_DATA_DIR` env var).

### 3. Hugging Face Token (recommended)

Set your HF token for faster downloads:

```bash
export HF_TOKEN=hf_your_token_here
```

---

## 🏃 Pipeline

### Quick start (full pipeline on Kaggle)

```bash
python run_pipeline.py --steps 0,1,2,3,4,5
```

### Step-by-step

| Step | Command | Description | GPU needed? | Internet needed? |
|------|---------|-------------|-------------|------------------|
| 0 | `python download_data.py --probe` | Probe HF dataset structure | ❌ | ✅ |
| 0 | `python download_data.py --download` | Download audio by ID | ❌ | ✅ |
| 1 | `python step1_preprocessing.py` | Folds + normalization + vocabs | ❌ | ❌ |
| 2 | `python step2_train_mms.py` | MMS-1B + LoRA fine-tuning | ✅ | ❌ |
| 3 | `python step3_train_kenlm.py` | KenLM 5-gram training | ❌ | ❌ |
| 3b | `python step3_decode_lm.py --model_dir ... --mode tune --fold 0` | Tune α/β on validation fold | ✅ | ❌ |
| 4 | `python step4_train_w2vbert.py` | w2v-BERT 2.0 + LoRA fine-tuning | ✅ | ❌ |
| 5 | `python step5_ensemble.py` | Ensemble + generate submission CSV | ✅ | ❌ |

### Selective runs

```bash
# Download audio + preprocess only:
python run_pipeline.py --steps 0,1

# Preprocess + train KenLM (no GPU needed):
python run_pipeline.py --steps 1,3

# Decode with tuned hyperparameters:
python step3_decode_lm.py --mode tune --fold 0

# Ensemble existing checkpoints:
python step5_ensemble.py --mms_model_dir ./output/checkpoints/mms --alpha 2.0 --beta 0.5
```

---

## 📥 Step 0: Data Download (`download_data.py`)

The full WAXAL dataset on Hugging Face is **1.06 TB** across 19 languages. This script downloads **only the audio files you need** (~6–15 GB) using streaming.

### Usage

```bash
# 1. First, probe the dataset to check the ID field mapping:
python download_data.py --probe

# 2. Then download all required audio:
python download_data.py --download

# Or download one language at a time (resume-friendly):
python download_data.py --download --lang lin
python download_data.py --download --lang sna
python download_data.py --download --lang lug
```

### How it works

1. Reads `Train.csv` and `Test.csv` → builds a set of ~42k required audio IDs
2. For each of the 3 languages, loads the HF dataset in **streaming mode** (no full download)
3. Matches each streamed sample against the required IDs
4. Saves matched audio as 16 kHz mono WAV files

### Flags

| Flag | Description |
|------|-------------|
| `--probe` | Show the HF dataset structure + ID field (run this first) |
| `--download` | Download audio matching competition IDs |
| `--lang {lin,sna,lug}` | Download only this language |
| `--data-dir PATH` | Custom data directory (default: `./data`) |

---

## 📊 Step 1: Preprocessing (`step1_preprocessing.py`)

Creates speaker-disjoint GroupKFold folds, normalizes transcripts, builds character vocabularies.

### What it produces

```
output/
├── train_folds.csv       # Training data with fold assignments
├── test_normalized.csv   # Normalized test transcripts
├── vocabs.json           # Per-language + combined character vocabs
└── all_transcripts.txt   # All normalized transcripts (for KenLM corpus)
```

### Why speaker-disjoint folds?

Phase 2 of the competition uses **completely unseen speakers** — no metadata, just raw audio. Random splits would let the model memorize speakers. GroupKFold ensures no speaker appears in both training and validation, giving a realistic estimate of generalization.

---

## 🎯 Step 2 & 4: Model Fine-Tuning

### MMS-1B (`step2_train_mms.py`)
- **Model:** `facebook/mms-1b-all` (1B params)
- **PEFT:** LoRA (`r=16`, `alpha=32`) on `q_proj`, `v_proj`, `out_proj`
- **SpecAugment:** frequency masking (27 masks) + time masking (10 masks)
- **Batch:** effective batch size = 16 (`per_device_batch=2`, `grad_accum=8`)
- **Trains 5 models** (one per fold) saved to `output/checkpoints/mms/fold_{i}/`

### w2v-BERT 2.0 (`step4_train_w2vbert.py`)
- **Model:** `facebook/w2v-bert-2.0`
- Same LoRA config and training schedule
- Trains 5 models saved to `output/checkpoints/w2v_bert/fold_{i}/`

> **Note:** Both steps require a GPU. On Kaggle, an A100 (free tier) can train one fold in ~2–3 hours.

---

## 📝 Step 3: Language Model

### KenLM training (`step3_train_kenlm.py`)
- Trains a 5-gram language model from `all_transcripts.txt`
- Requires `lmplz` and `build_binary` (installed via `pip install kenlm`)
- Output: `output/kenlm/waxal_5gram.arpa` + binary

### Hyperparameter tuning (`step3_decode_lm.py --mode tune`)
- Grid search over alpha (LM weight) and beta (word insertion bonus)
- Uses pyctcdecode for CTC beam search
- Evaluates on fold 0 to find optimal (α, β)
- Results saved to `output/kenlm/tune_results.json`

### Decoding (`step3_decode_lm.py --mode decode`)
- Generates predictions using a trained model + tuned LM
- Supports per-language decoders or a combined decoder
- Saves predictions as JSON for ensemble step

---

## 🔗 Step 5: Ensemble (`step5_ensemble.py`)

- Loads MMS-1B and w2v-BERT 2.0 checkpoints
- Averages frame-level logits: `0.5 × logits_mms + 0.5 × logits_w2v`
- Decodes with KenLM using tuned α/β
- Generates submission CSV in `output/submissions/`

---

## 📐 Configuration (`config.py`)

All hyperparameters are centralized in `config.py`. Key things to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_DIR` | `./data` | Where CSVs + audio live |
| `N_FOLDS` | `5` | Cross-validation folds |
| `LANGUAGES` | `["Lingala", "Shona", "Luganda"]` | Target languages |
| `MMSConfig.lora_r` | `16` | LoRA rank |
| `KenLMConfig.ngram_order` | `5` | N-gram order |
| `BeamSearchConfig.alpha_range` | `[0, 0.5, 1, 1.5, 2, 2.5, 3]` | LM weight candidates |
| `BeamSearchConfig.beta_range` | `[0, 0.5, 1, 1.5, 2]` | Word bonus candidates |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `WAXAL_DATA_DIR` | Override data directory |
| `HF_TOKEN` | Hugging Face token (faster downloads) |
| `HF_USERNAME` | Hugging Face username (for pushing models) |

---

## 📦 Output Structure

```
output/
├── pretrained/                # Downloaded base models (cached)
├── checkpoints/
│   ├── mms/fold_{0-4}/        # MMS-1B LoRA adapters
│   └── w2v_bert/fold_{0-4}/   # w2v-BERT 2.0 LoRA adapters
├── kenlm/
│   ├── waxal_5gram.arpa       # KenLM ARPA model
│   ├── waxal_5gram.bin        # KenLM binary (faster loading)
│   └── tune_results.json      # Alpha/beta tuning results
├── submissions/
│   └── submission_*.csv       # Final submission files
├── train_folds.csv
├── test_normalized.csv
├── vocabs.json
└── all_transcripts.txt
```

---

## ⚠️ Important Notes

### Phase 1 vs Phase 2
- **Phase 1:** You get labeled train/validation + unlabeled test audio. Leaderboard is for development only.
- **Phase 2:** A completely new test set with **no metadata** (language, speaker, gender) is released ~1 week before close. **Final rankings** are based on Phase 2 performance only.
- **Do not** use Phase 1 test set ground-truth labels for training — this breaches competition rules.

### Disk space
- Full WAXAL dataset: **1.06 TB** (do not download)
- Our selective download: **~6–15 GB** (safe for Kaggle free tier, limit: 57.6 GiB)
- After training + checkpoints: ~20–30 GB total

### GPU memory
- MMS-1B (1B params) + LoRA: ~16 GB VRAM with batch size 2
- w2v-BERT 2.0 (~600M params): ~12 GB VRAM with batch size 2
- Kaggle A100 (40 GB) is sufficient

### Common issues
- **`pyenv: shell integration not enabled`** — harmless, just a warning. The scripts still run correctly.
- **`lmplz: command not found`** — install kenlm: `pip install kenlm && python -c "import kenlm"` (builds the binaries)
- **`numpy` version conflict** — pyctcdecode wants `numpy<2` but `numpy>=2` is installed. Works in practice but may warn. Downgrade if you hit issues.
