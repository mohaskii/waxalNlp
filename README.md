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

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/waxalNlp.git
cd waxalNlp
pip install -r requirements.txt
```

**Key packages:**
| Package | Purpose |
|---|---|
| `transformers`, `peft` | HuggingFace model + LoRA |
| `datasets` | HF dataset streaming |
| `torch` | Deep learning |
| `pyctcdecode`, `kenlm` | CTC beam search + language model |
| `librosa`, `soundfile` | Audio loading |
| `jiwer` | WER / CER metrics |
| `scikit-learn` | GroupKFold cross-validation |

### 2. Data layout

The competition provides CSV files; audio is fetched from Hugging Face.

```
data/                          # CSVs go here (tiny, ~8 MB total)
├── Train.csv
├── Test.csv
└── SampleSubmission.csv

/tmp/data/                     # Audio goes here on Kaggle (~18 GB as FLAC)
├── Train/
│   ├── lug_96123.flac
│   ├── lin_4521.flac
│   └── ...
└── Test/
    ├── lug_96114.flac
    └── ...
```

> **Why /tmp?** On Kaggle, `/tmp` does **not** count toward the 20 GB output limit.
> Audio is re-downloaded each session (it takes ~40 minutes for all 3 languages).

### 3. Hugging Face token

```bash
export HF_TOKEN="hf_your_read_token_here"
```

Create a read-only token at https://huggingface.co/settings/tokens. The dataset `google/WaxalNLP` is public — this just avoids rate limits.

### 4. Kaggle GPU note

Training requires a GPU. On Kaggle:
- **Accelerator:** GPU T4 x2 (or A100 if available)
- **Internet:** ON (needed for dataset download + model weights)

---

## ⏱️ Quick Start: MMS-Only Submission (23h)

**Best for:** Limited GPU time (~23 hours/week), first submission on the leaderboard.

### Session plan

| Session | What | Command | Time |
|---|---|---|---|
| 1 | Download + preprocess + KenLM + tune | `download_data.py --download` then `step1_preprocessing.py` then `step3_train_kenlm.py` then `step3_decode_lm.py --mode tune --fold 0` | ~1h |
| 2 | Train MMS fold 0 | `step2_train_mms.py --fold 0` | ~4.5h |
| 3 | Train MMS fold 1 | `step2_train_mms.py --fold 1` | ~4.5h |
| 4 | Train MMS fold 2 | `step2_train_mms.py --fold 2` | ~4.5h |
| 5 | Train MMS fold 3 | `step2_train_mms.py --fold 3` | ~4.5h |
| 6 | Decode + submit | `step5_ensemble.py --single_model mms --mms_model_dir output/checkpoints/mms_fold_0 --alpha X --beta Y` | ~30m |

**Total: ~19.5h** (3.5h buffer for variance).

### What this gives you

- 4 trained MMS-1B LoRA adapters (fold 0–3), each auto-pushed to HF Hub for safety
- One fold decoded with KenLM + tuned α/β → submission CSV
- No w2v-BERT training (saves 20h of GPU time)
- No multi-fold ensemble (easily added later if time permits)

### Key config changes for this plan

```python
# In config.py:
num_train_epochs: int = 5   # was 10 — saves 5h per fold
push_to_hub: bool = True    # auto-save each fold to HF Hub
```

### Adding ensemble later (if time permits)

```bash
# Train remaining fold:
python step2_train_mms.py --fold 4

# Ensemble all 5 folds (requires modifying step5 to average across folds):
python step5_ensemble.py --single_model mms \
  --mms_model_dir output/checkpoints/mms_fold_0 \
  --alpha X --beta Y
```

> **Why 5 epochs?** LoRA converges fast. 5 epochs × 4 folds = better ensemble diversity than 10 epochs × 2 folds. More folds > more epochs for this competition's unseen-speaker test set.

---

## 🏃 Full Pipeline (Step by Step)

### Quick reference table

| Step | Script | GPU? | Internet? | Time (est.) | Output |
|------|--------|------|-----------|-------------|--------|
| 0 | `download_data.py --download` | ❌ | ✅ | ~40 min | `/tmp/data/{Train,Test}/*.flac` |
| 1 | `step1_preprocessing.py` | ❌ | ❌ | <1 min | `output/train_folds.csv`, `vocabs.json` |
| 2 | `step2_train_mms.py` | ✅ | ❌ | ~3h/fold×5 | `output/checkpoints/mms/` |
| 3 | `step3_train_kenlm.py` | ❌ | ❌ | <5 min | `output/kenlm/waxal_5gram.bin` |
| 3b | `step3_decode_lm.py --mode tune` | ✅ | ❌ | ~15 min | `output/kenlm/tune_results.json` |
| 4 | `step4_train_w2vbert.py` | ✅ | ❌ | ~2h/fold×5 | `output/checkpoints/w2v_bert/` |
| 5 | `step5_ensemble.py` | ✅ | ❌ | ~10 min | `output/submissions/submission_*.csv` |

### Orchestrated run

```bash
# Everything from scratch (on Kaggle):
python run_pipeline.py --steps 0,1,2,3,4,5

# Selective:
python run_pipeline.py --steps 0,1        # download + preprocess
python run_pipeline.py --steps 1,3        # preprocess + KenLM (no GPU)
python run_pipeline.py --steps 5          # ensemble only (existing checkpoints)
```

---

## 📥 Step 0: Download Audio (`download_data.py`)

### Why selective download?

| | Full HF dataset | What we download |
|---|---|---|
| Languages | 19 | 3 (lin, sna, lug) |
| Size | 1.06 TB | ~18 GB (FLAC) |
| Method | Raw download | Streaming, ID-matched |

The script reads `Train.csv` / `Test.csv`, builds a list of 42k required IDs, then streams the HF dataset and saves only matching files.

### Usage

```bash
# 1. Probe the dataset structure (verify ID field)
python download_data.py --probe

# 2. Download all audio
python download_data.py --download

# 3. Download one language at a time (resume-friendly)
python download_data.py --download --lang lin
python download_data.py --download --lang sna
python download_data.py --download --lang lug
```

### Flags

| Flag | Description |
|---|---|
| `--probe` | Show dataset structure + ID field (no download) |
| `--download` | Download matching audio |
| `--lang {lin,sna,lug}` | Single language only |
| `--audio-dir PATH` | Where to save audio (default: auto-detected) |
| `--csv-dir PATH` | Where to find Train.csv / Test.csv (default: `./data`) |

### Format

Audio is saved as **int16 FLAC** (lossless, ~50% of WAV size). `librosa.load()` reads FLAC transparently — no code changes needed in training scripts.

### Disk space on Kaggle

| Component | Size |
|---|---|
| Python packages (torch, transformers, ...) | ~15 GB |
| Audio files (42k × int16 FLAC) | ~18 GB |
| Checkpoints (2 models × 5 folds) | ~10 GB |
| **Total** | **~43 GB** |

The VM limit is ~57.6 GB — you have room. Audio goes to `/tmp/` (doesn't count toward the 20 GB **output** limit), so checkpoints and submissions fit in `/kaggle/working/output/`.

### "I ran out of space!"

```bash
# Delete old audio and re-download as FLAC:
rm -rf /tmp/data/
python download_data.py --download
```

---

## 📊 Step 1: Preprocessing (`step1_preprocessing.py`)

```bash
python step1_preprocessing.py
```

### What it does

| Action | Why |
|---|---|
| **Speaker-disjoint GroupKFold** (5 folds) | Phase 2 tests on unseen speakers. GroupKFold forces generalization across speakers, not memorization. |
| **Unicode NFC normalization** | African languages use accented characters (e.g., `ɛ́`, `ɔ́`, `ŋ`). NFC collapses multi-codepoint forms into single codepoints — prevents the model from seeing the same character as two different tokens. |
| **Per-language character vocabularies** | Builds `{char: index}` mappings for Lingala, Shona, Luganda individually + combined. Used by the CTC decoder to know which characters to output. |
| **Save artifacts** | Produces `train_folds.csv`, `test_normalized.csv`, `vocabs.json`, `all_transcripts.txt` in `output/`. |

### Outputs

```
output/
├── train_folds.csv          # All training data with fold 0-4 assignments
├── test_normalized.csv      # Test data with normalized transcripts
├── vocabs.json              # {"lin": {char: idx, ...}, "sna": {...}, "combined": {...}}
└── all_transcripts.txt      # One transcript per line (KenLM corpus)
```

---

## 🎯 Step 2 & 4: Model Fine-Tuning

### MMS-1B (`step2_train_mms.py`)

```bash
python step2_train_mms.py
```

| Setting | Value | Why |
|---|---|---|
| Base model | `facebook/mms-1b-all` | Pre-trained on 1,200+ languages |
| LoRA rank | `r=16, alpha=32` | Lightweight adapters — fits in A100 VRAM |
| Target modules | `q_proj, v_proj, out_proj` | Attention projections only |
| Effective batch | 16 (2 × 8 grad accum) | Maximizes GPU utilization |
| SpecAugment | freq=27, time=10 | Prevents overfitting on acoustic features |
| Epochs | 10 | With LoRA, converges quickly |
| Folds | 5 | One model per fold for ensemble |

### w2v-BERT 2.0 (`step4_train_w2vbert.py`)

```bash
python step4_train_w2vbert.py
```

| Setting | Value | Why |
|---|---|---|
| Base model | `facebook/w2v-bert-2.0` | Conformer-based — architecturally different from MMS for better ensemble diversity |
| LoRA config | Same as MMS | Consistent training regime |
| Epochs | 10 | Matches MMS schedule |

> **Training both gives you ensemble diversity.** MMS uses a Wav2Vec2 architecture (CNN encoder + Transformer), w2v-BERT uses Conformer (CNN + Conformer blocks). Averaging their logits produces a stronger combined prediction.

### Checkpoint structure

```
output/checkpoints/
├── mms/
│   ├── fold_0/   # Trained on folds 1-4, validated on fold 0
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
└── w2v_bert/
    ├── fold_0/
    ├── ...
    └── fold_4/
```

---

## 📝 Step 3: Language Model

### Train KenLM (`step3_train_kenlm.py`)

```bash
python step3_train_kenlm.py
```

- Reads `output/all_transcripts.txt` (all normalized training transcripts)
- Builds a **5-gram KenLM model** with pruning `[0, 1, 2]`
- Saves ARPA + binary to `output/kenlm/`

### Tune hyperparameters (`step3_decode_lm.py`)

```bash
python step3_decode_lm.py --mode tune --fold 0
```

Performs grid search over:

| Parameter | Meaning | Range |
|---|---|---|
| α (alpha) | Language model weight | 0.0 – 3.0 |
| β (beta) | Word insertion bonus | 0.0 – 2.0 |

Uses **pyctcdecode** for CTC beam search with beam width 100. Evaluates WER + CER on the validation fold. Saves results to `output/kenlm/tune_results.json`.

Output example:
```
Best: α=1.5, β=0.5  (combined=0.2341)
```

### Decode (generate predictions)

```bash
python step3_decode_lm.py --mode decode --alpha 1.5 --beta 0.5
```

---

## 🔗 Step 5: Ensemble (`step5_ensemble.py`)

```bash
# With tuned hyperparameters from step 3b:
python step5_ensemble.py --alpha 1.5 --beta 0.5

# With custom paths:
python step5_ensemble.py \
    --mms_model_dir ./output/checkpoints/mms \
    --w2v_model_dir ./output/checkpoints/w2v_bert \
    --alpha 1.5 --beta 0.5
```

### How it works

1. Loads MMS-1B + w2v-BERT checkpoints for each fold
2. Runs each audio sample through **both models**
3. Averages frame-level logits: `0.5 × logits_mms + 0.5 × logits_w2v`
4. Decodes with pyctcdecode + KenLM using the tuned α/β
5. Generates `submission.csv` in `output/submissions/`

### Why ensemble?

| Model | Architecture | Trained on |
|---|---|---|
| MMS-1B | Wav2Vec2 (CNN + Transformer) | Folds 1-4 |
| w2v-BERT 2.0 | Conformer (CNN + Conformer) | Folds 1-4 |

Different architectures make different types of errors. Averaging their predictions produces a more robust output — typically reducing WER by **5–15%** over either model alone.

---

## 📐 Configuration (`config.py`)

Customize by editing `config.py` or setting environment variables:

### Key settings

| Setting | Default | Description |
|---|---|---|
| `N_FOLDS` | 5 | Cross-validation folds |
| `LANGUAGES` | `["Lingala", "Shona", "Luganda"]` | Target languages |
| `MAX_AUDIO_LENGTH` | 30.0 | Truncate audio longer than this (seconds) |
| `MMSConfig.lora_r` | 16 | LoRA adapter rank |
| `MMSConfig.lora_alpha` | 32 | LoRA scaling factor |
| `MMSConfig.num_train_epochs` | 10 | Training epochs |
| `MMSConfig.per_device_train_batch_size` | 2 | Batch size per GPU |
| `KenLMConfig.ngram_order` | 5 | N-gram size |
| `BeamSearchConfig.beam_width` | 100 | CTC beam width |
| `BeamSearchConfig.alpha_range` | `[0, 0.5, ..., 3.0]` | LM weight candidates for tuning |
| `ENSEMBLE_WEIGHTS` | `{"mms": 0.5, "w2v_bert": 0.5}` | Logit averaging weights |

### Environment variables

| Variable | Purpose |
|---|---|
| `WAXAL_DATA_DIR` | Override audio storage directory |
| `HF_TOKEN` | HuggingFace token (faster downloads) |
| `HF_USERNAME` | HuggingFace username (for pushing models) |

---

## 📦 Complete Output Structure

```
output/
├── pretrained/                    # Cached base models
├── checkpoints/
│   ├── mms/fold_{0-4}/            # MMS-1B LoRA adapters
│   └── w2v_bert/fold_{0-4}/       # w2v-BERT LoRA adapters
├── kenlm/
│   ├── waxal_5gram.arpa           # KenLM ARPA (text)
│   ├── waxal_5gram.bin            # KenLM binary (fast)
│   └── tune_results.json          # Alpha/beta + all metrics
├── submissions/
│   └── submission_*.csv           # Ready for Zindi upload
├── train_folds.csv
├── test_normalized.csv
├── vocabs.json
└── all_transcripts.txt

/tmp/data/                         # Audio files (session-only on Kaggle)
├── Train/  →  *.flac
└── Test/   →  *.flac
```

---

## ⚠️ Important Notes

### Phase 1 vs Phase 2

| | Phase 1 | Phase 2 |
|---|---|---|
| **Data** | Labeled train/val + unlabeled test audio | Brand new unseen test set |
| **Metadata** | Language + speaker IDs provided | **No metadata** — just raw audio |
| **Leaderboard** | Development / experimentation | **Final rankings + prizes** |
| **Timing** | Now | ~1 week before challenge closes |

**Do NOT** use Phase 1 test set ground-truth labels for training. Breaches competition rules → disqualification.

### Why speaker-disjoint folds matter

If you use random splits, the same speaker appears in both training and validation. The model memorizes their voice, validation WER looks great, but it fails on Phase 2's unseen speakers. GroupKFold forces your validation metric to reflect **real generalization**.

### Common issues

| Symptom | Fix |
|---|---|
| `pyenv: shell integration not enabled` | Harmless warning, ignore |
| `RepositoryNotFoundError: 401` on HF | HF token expired. Create new read-only token at huggingface.co/settings/tokens |
| `User Access Token "lolo" is expired` | Kaggle has a stale cached token. Set `HF_TOKEN` in your environment or config.py |
| `ModuleNotFoundError: No module named 'datasets'` | Run `pip install datasets` (already installed on Kaggle) |
| Ran out of disk space | Audio went to `/kaggle/working/` instead of `/tmp/`. Delete + re-download |
| `lmplz: command not found` | Run `pip install kenlm` (builds the binaries) |
| `numpy` version warning from pyctcdecode | `pyctcdecode` prefers `numpy<2`. Works with `numpy≥2` in practice |
| CUDA out of memory | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` to 16 |
