"""
Central configuration for the WAXAL ASR Challenge pipeline.
All paths, hyperparameters, and model settings live here.
"""

import os
from dataclasses import dataclass, field

os.environ["HF_TOKEN"] = "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"

# ---------------------------------------------------------------------------
# Data Paths (adjust to your Kaggle / local environment)
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("WAXAL_DATA_DIR", "./data")
TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
TEST_CSV = os.path.join(DATA_DIR, "Test.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "Train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "Test")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "SampleSubmission.csv")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output"
PRETRAINED_DIR = os.path.join(OUTPUT_DIR, "pretrained")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
KENLM_DIR = os.path.join(OUTPUT_DIR, "kenlm")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")
os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(KENLM_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------
HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# WAXAL dataset on Hugging Face (for selective audio download)
# Full dataset: 1.06 TB across 19 languages
# We download only the 3 competition languages via streaming
HF_DATASET_ID = "google/WaxalNLP"
# Mapping: competition language code -> HF dataset config name
LANG_TO_HF_CONFIG = {
    "lin": "lin_asr",
    "sna": "sna_asr",
    "lug": "lug_asr",
}

# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLING_RATE = 16000
MAX_AUDIO_LENGTH = 30.0  # seconds; clip or pad longer utterances

# ---------------------------------------------------------------------------
# Step 1: Preprocessing & CV
# ---------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42
CV_METRIC_WEIGHTS = {"wer": 0.5, "cer": 0.5}


# ---------------------------------------------------------------------------
# Step 2: MMS-1B Fine-Tuning
# ---------------------------------------------------------------------------
@dataclass
class MMSConfig:
    model_name: str = "facebook/mms-1b-all"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "out_proj"]
    )
    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    num_train_epochs: int = 10
    fp16: bool = True
    # SpecAugment
    specaug_freq_mask: int = 27
    specaug_time_mask: int = 10
    # Logging & saving
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    push_to_hub: bool = True
    hub_model_id: str = "mms-1b-waxal"


# ---------------------------------------------------------------------------
# Step 3: KenLM
# ---------------------------------------------------------------------------
@dataclass
class KenLMConfig:
    ngram_order: int = 5
    prune_values: list[int] = field(default_factory=lambda: [0, 1, 2])


# ---------------------------------------------------------------------------
# Step 3b: CTC Beam Search Decoding (pyctcdecode)
# ---------------------------------------------------------------------------
@dataclass
class BeamSearchConfig:
    alpha_range: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    beta_range: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    beam_width: int = 100


# ---------------------------------------------------------------------------
# Step 4: w2v-BERT 2.0 Fine-Tuning
# ---------------------------------------------------------------------------
@dataclass
class W2VBertConfig:
    model_name: str = "facebook/w2v-bert-2.0"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "out_proj"]
    )
    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    num_train_epochs: int = 10
    fp16: bool = True
    specaug_freq_mask: int = 27
    specaug_time_mask: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    push_to_hub: bool = True
    hub_model_id: str = "w2v-bert-2.0-waxal"


# ---------------------------------------------------------------------------
# Step 5: Ensembling
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {"mms": 0.5, "w2v_bert": 0.5}

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
LANGUAGES = ["Lingala", "Shona", "Luganda"]
