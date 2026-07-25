"""
Shared utilities for the WAXAL ASR pipeline: audio loading, text normalization,
vocabulary building, and metric computation.
"""

from __future__ import annotations

import os
import re
import unicodedata

import librosa
import numpy as np
import torch
from jiwer import cer, wer

import config


# ---------------------------------------------------------------------------
# Text Normalization (Step 1.2 in the master plan)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """
    Apply Unicode NFC normalization, lowercase, strip non-standard punctuation.
    This is critical for CER performance on accented characters in
    Lingala, Shona, and Luganda.
    """
    # Unicode NFC normalization — handles composed accented characters
    text = unicodedata.normalize("NFC", text).strip()
    # Lowercase
    text = text.lower()
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Vocabulary Building (Step 1.3)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<pad>", "<unk>", "<space>", "<blank>"]


def build_vocab(transcripts: list[str]) -> dict[str, int]:
    """
    Extract all unique characters from normalized transcripts and build
    a CTC-compatible vocabulary dict including special tokens.
    Returns: { char: index } ordered dict.
    """
    chars = set()
    for t in transcripts:
        chars.update(t)
    # Sort for determinism
    sorted_chars = sorted(chars)
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for i, ch in enumerate(sorted_chars, start=len(SPECIAL_TOKENS)):
        vocab[ch] = i
    return vocab


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------
def resolve_audio_path(path: str) -> str:
    """Try path as-is, then .flac, then .wav."""
    if os.path.exists(path):
        return path
    for ext in (".flac", ".wav"):
        candidate = path + ext
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Audio not found: {path}")


def load_audio(
    path: str, target_sr: int = config.SAMPLING_RATE
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Load an audio file and resample to target_sr. Returns float32 numpy array."""
    resolved = resolve_audio_path(path)
    audio, _ = librosa.load(resolved, sr=target_sr, mono=True)
    # Clip to MAX_AUDIO_LENGTH
    max_samples = int(config.MAX_AUDIO_LENGTH * target_sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32)


def load_audio_for_model(path: str, feature_extractor) -> dict[str, torch.Tensor]:
    """
    Load audio and pass through the model's feature extractor.
    Returns the processed input_values tensor and attention_mask.
    """
    audio = load_audio(path)
    return feature_extractor(
        audio,
        sampling_rate=config.SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred_strs: list[str], label_strs: list[str]) -> dict[str, float]:
    """Compute WER, CER, and the combined CV metric."""
    w = wer(label_strs, pred_strs)
    c = cer(label_strs, pred_strs)
    combined = config.CV_METRIC_WEIGHTS["wer"] * w + config.CV_METRIC_WEIGHTS["cer"] * c
    return {"wer": w, "cer": c, "combined": combined}


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC that dynamically pads inputs to the longest
    sequence in the batch.
    """

    def __init__(self, feature_extractor, tokenizer, padding: bool = True):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.feature_extractor.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # Replace padding with -100 so CTC loss ignores it
        labels = labels_batch["input_values"].masked_fill(
            labels_batch["input_values"] == self.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels

        return batch


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected — training will be slow!")
