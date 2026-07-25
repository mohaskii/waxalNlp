"""
Step 5: Ensembling & Post-Processing
=====================================
- Frame-level logit averaging between MMS-1B and w2v-BERT 2.0
- Decode with KenLM CTC beam search
- Unicode NFC normalization and space trimming
- Generate final submission CSV

Architecture:
             [ Test Audio Frame ]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
[ Meta MMS-1B Logits ]  [ w2v-BERT 2.0 Logits ]
         │                         │
         └────────────┬────────────┘
                      ▼
        [ Average Frame Probabilities ]
                      │
                      ▼
    [ CTC Beam Search + KenLM Decoding ]
                      │
                      ▼
         [ Unicode NFC Normalization ]
                      │
                      ▼
           [ Final Submission CSV ]
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm

import config
from step3_decode_lm import load_model_and_processor
from step3_train_kenlm import load_audio_for_model
from utils import normalize_text


# ---------------------------------------------------------------------------
# Logit Extraction
# ---------------------------------------------------------------------------
def extract_logits(model, processor, audio_path: str) -> np.ndarray:
    """
    Run a single audio file through the model and return frame-level logits.
    Shape: (time_steps, vocab_size)
    """
    device = next(model.parameters()).device
    inputs = load_audio_for_model(audio_path, processor)
    with torch.no_grad():
        outputs = model(inputs["input_values"].to(device))
    return outputs.logits[0].cpu().numpy()


def average_logits(logits_list: list[np.ndarray], weights: list | None = None) -> np.ndarray:
    """
    Average multiple logit matrices with optional weights.
    All logits must have the same vocabulary size.  Shorter ones are padded
    to the maximum time dimension.
    """
    if weights is None:
        weights = [1.0 / len(logits_list)] * len(logits_list)

    max_time = max(logits.shape[0] for logits in logits_list)
    vocab_size = logits_list[0].shape[1]

    # Sanity check: all must have the same vocab size
    for i, l in enumerate(logits_list):
        assert l.shape[1] == vocab_size, (
            f"Model {i} vocab size {l.shape[1]} != {vocab_size}"
        )

    summed = np.zeros((max_time, vocab_size), dtype=np.float64)
    count = np.zeros((max_time, 1), dtype=np.float64)

    for logits, w in zip(logits_list, weights):
        t = logits.shape[0]
        summed[:t] += w * logits.astype(np.float64)
        count[:t] += w

    # Avoid division by zero
    count[count == 0] = 1.0
    return (summed / count).astype(np.float32)


# ---------------------------------------------------------------------------
# Ensemble Decoding Pipeline
# ---------------------------------------------------------------------------
def ensemble_decode(
    model1, processor1,
    model2, processor2,
    decoder,
    test_df: pd.DataFrame,
    weights: tuple | None = None,
) -> list[str]:
    """
    For every test audio:
        1. Extract logits from both models
        2. Average frame probabilities
        3. Beam search decode with KenLM
        4. NFC normalize
    Returns list of final transcripts.
    """
    if weights is None:
        weights = (0.5, 0.5)

    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Ensembling"):
        audio_path = os.path.join(config.TEST_AUDIO_DIR, str(row["Audio_ID"]))
        if not os.path.exists(audio_path):
            print(f"WARNING: {audio_path} not found")
            predictions.append("")
            continue

        # Extract logits from both models
        logits1 = extract_logits(model1, processor1, audio_path)
        logits2 = extract_logits(model2, processor2, audio_path)

        # Average
        averaged = average_logits([logits1, logits2], list(weights))

        # Beam search decode
        text = decoder.decode(averaged)

        # Post-process: NFC normalization + space trimming
        text = normalize_text(text)
        predictions.append(text)

    return predictions


# ---------------------------------------------------------------------------
# Single Model Decode (for comparison)
# ---------------------------------------------------------------------------
def single_model_decode(
    model, processor, decoder, test_df: pd.DataFrame
) -> list[str]:
    """Decode test set with a single model + KenLM beam search."""
    device = next(model.parameters()).device
    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Single model"):
        audio_path = os.path.join(config.TEST_AUDIO_DIR, str(row["Audio_ID"]))
        if not os.path.exists(audio_path):
            predictions.append("")
            continue

        inputs = load_audio_for_model(audio_path, processor)
        with torch.no_grad():
            logits = model(inputs["input_values"].to(device)).logits[0].cpu().numpy()
        text = decoder.decode(logits)
        predictions.append(normalize_text(text))

    return predictions


# ---------------------------------------------------------------------------
# Build Decoder from Vocab + KenLM
# ---------------------------------------------------------------------------
def build_beam_decoder(vocab_path: str, kenlm_path: str, alpha: float, beta: float):
    """Build the pyctcdecode beam search decoder with KenLM."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)
    vocab = vocabs["combined"]
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]

    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Inference & Submission")
    parser.add_argument("--mms_model_dir", type=str, default=None,
                        help="Path to MMS-1B checkpoint (with lora_adapter/)")
    parser.add_argument("--w2v_model_dir", type=str, default=None,
                        help="Path to w2v-BERT 2.0 checkpoint (with lora_adapter/)")
    parser.add_argument("--kenlm_path", type=str,
                        default=os.path.join(config.KENLM_DIR, "lm_5gram.binary"))
    parser.add_argument("--vocab_path", type=str,
                        default=os.path.join(config.OUTPUT_DIR, "vocabs.json"))
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mms_weight", type=float, default=0.5)
    parser.add_argument("--w2v_weight", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str,
                        default=os.path.join(config.SUBMISSION_DIR, "ensemble_final.csv"))
    parser.add_argument("--single_model", type=str, default=None,
                        choices=["mms", "w2v", None],
                        help="If set, run single-model decode instead of ensemble")

    args = parser.parse_args()

    print("=" * 60)
    print("STEP 5: Ensembling & Final Submission")
    print("=" * 60)

    # Load test data
    test_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "test_normalized.csv"))
    print(f"Test samples: {len(test_df)}")

    # Build decoder
    print("\nBuilding KenLM CTC beam search decoder...")
    decoder = build_beam_decoder(args.vocab_path, args.kenlm_path, args.alpha, args.beta)
    print(f"Decoder ready (α={args.alpha}, β={args.beta})")

    if args.single_model:
        # Single model decode
        model_dir = args.mms_model_dir if args.single_model == "mms" else args.w2v_model_dir
        print(f"\nSingle model decode: {args.single_model}")
        model, processor = load_model_and_processor(model_dir)
        predictions = single_model_decode(model, processor, decoder, test_df)
    else:
        # Ensemble decode
        print(f"\nLoading MMS-1B from {args.mms_model_dir}...")
        model1, processor1 = load_model_and_processor(args.mms_model_dir)

        print(f"Loading w2v-BERT 2.0 from {args.w2v_model_dir}...")
        model2, processor2 = load_model_and_processor(args.w2v_model_dir)

        weights = (args.mms_weight, args.w2v_weight)
        print(f"Ensemble weights: MMS={weights[0]}, w2v-BERT={weights[1]}")

        predictions = ensemble_decode(
            model1, processor1,
            model2, processor2,
            decoder, test_df, weights,
        )

    # Final post-processing pass (NFC normalization already done in ensemble_decode)
    predictions = [normalize_text(p) for p in predictions]

    # Create submission
    submission = pd.DataFrame({
        "Audio_ID": test_df["Audio_ID"],
        "Predicted_Transcript": predictions,
    })

    submission.to_csv(args.output_csv, index=False)
    print(f"\nSubmission saved to {args.output_csv}")
    print(f"  Rows: {len(submission)}")
    print(f"  Sample: {predictions[0][:80] if predictions else 'N/A'}")

    print("\nStep 5 complete! Submit this CSV to Zindi.")


if __name__ == "__main__":
    main()
