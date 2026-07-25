"""
Step 3b: CTC Beam Search Decoding with KenLM
=============================================
Load a fine-tuned model and use pyctcdecode's CTC Beam Search
with the trained KenLM to decode audio.  Also provides the
hyperparameter tuning entry point.
"""

import json
import os

import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm

import config
from step3_train_kenlm import load_audio_for_model
from utils import compute_metrics, normalize_text


def load_model_and_processor(model_dir: str):
    """
    Load a Wav2Vec2ForCTC model with LoRA adapters and its processor.
    """
    from peft import PeftModel
    from transformers import AutoModelForCTC, AutoProcessor

    # If we have a LoRA adapter directory, load base + adapter
    adapter_path = os.path.join(model_dir, "lora_adapter")
    if os.path.isdir(adapter_path):
        # Determine base model from adapter config
        base_model_name = "facebook/mms-1b-all"  # default; could read from config
        model = AutoModelForCTC.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        processor = AutoProcessor.from_pretrained(
            base_model_name, trust_remote_code=True
        )
    else:
        model = AutoModelForCTC.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, processor


def build_beam_decoder(
    vocab_path: str, kenlm_path: str, alpha: float = 1.5, beta: float = 1.0
):
    """Build a pyctcdecode CTC beam search decoder."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)
    vocab = vocabs["combined"]

    # Build label list in index order
    # vocab is {char: index}; need [char0, char1, ...]
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta,
    )
    return decoder


def decode_dataset(
    model,
    processor,
    decoder,
    df: pd.DataFrame,
    audio_dir: str,
    batch_size: int = 1,
) -> list[str]:
    """
    Decode all audio files in the dataframe using CTC beam search + KenLM.
    Returns list of predicted transcripts.
    """
    device = next(model.parameters()).device
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Decoding"):
        audio_path = os.path.join(audio_dir, str(row["Audio_ID"]))
        if not os.path.exists(audio_path):
            print(f"WARNING: {audio_path} not found")
            predictions.append("")
            continue

        inputs = load_audio_for_model(audio_path, processor)
        with torch.no_grad():
            outputs = model(inputs["input_values"].to(device))
        logits = outputs.logits[0].cpu().numpy()  # (time, vocab_size)
        text = decoder.decode(logits)
        predictions.append(normalize_text(text))

    return predictions


def tune_on_validation_fold(
    model,
    processor,
    val_df: pd.DataFrame,
    vocabs: dict,
    kenlm_path: str,
    beam_cfg: config.BeamSearchConfig,
):
    """
    Full hyperparameter grid search for alpha and beta on a validation fold.
    """
    import itertools

    device = next(model.parameters()).device
    references = val_df["Transcript_normalized"].tolist()
    vocab = vocabs["combined"]
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]

    best_combined = float("inf")
    best_params = (1.0, 1.0)

    print(
        f"\nGrid search over alpha {beam_cfg.alpha_range}, beta {beam_cfg.beta_range}"
    )

    for alpha, beta in itertools.product(beam_cfg.alpha_range, beam_cfg.beta_range):
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_path,
            alpha=alpha,
            beta=beta,
        )
        predictions = []

        for _, row in val_df.iterrows():
            audio_path = os.path.join(config.TRAIN_AUDIO_DIR, str(row["Audio_ID"]))
            inputs = load_audio_for_model(audio_path, processor)
            with torch.no_grad():
                logits = (
                    model(inputs["input_values"].to(device)).logits[0].cpu().numpy()
                )
            predictions.append(normalize_text(decoder.decode(logits)))

        metrics = compute_metrics(predictions, references)
        print(
            f"  α={alpha:.1f} β={beta:.1f}  "
            f"WER={metrics['wer']:.4f} CER={metrics['cer']:.4f} "
            f"C={metrics['combined']:.4f}"
        )

        if metrics["combined"] < best_combined:
            best_combined = metrics["combined"]
            best_params = (alpha, beta)

    print(f"\nBest α={best_params[0]}, β={best_params[1]} (C={best_combined:.4f})")
    return best_params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CTC Beam Search Decoding with KenLM")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--kenlm_path",
        type=str,
        default=os.path.join(config.KENLM_DIR, "lm_5gram.binary"),
        help="Path to KenLM binary model",
    )
    parser.add_argument(
        "--vocab_path", type=str, default=os.path.join(config.OUTPUT_DIR, "vocabs.json")
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "tune"],
        help="predict = decode test set; tune = grid search α/β on val fold",
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="Validation fold to use for tuning"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(config.SUBMISSION_DIR, "mms_beam.csv"),
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3b: CTC Beam Search Decoding")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    model, processor = load_model_and_processor(args.model_dir)

    if args.mode == "predict":
        # Decode test set
        test_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "test_normalized.csv"))
        decoder = build_beam_decoder(
            args.vocab_path, args.kenlm_path, args.alpha, args.beta
        )
        predictions = decode_dataset(
            model, processor, decoder, test_df, config.TEST_AUDIO_DIR
        )

        # Create submission
        submission = pd.DataFrame(
            {
                "Audio_ID": test_df["Audio_ID"],
                "Predicted_Transcript": predictions,
            }
        )
        submission.to_csv(args.output_csv, index=False)
        print(f"Submission saved to {args.output_csv}")

    elif args.mode == "tune":
        # Tune on validation fold
        train_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "train_folds.csv"))
        val_df = train_df.loc[train_df["fold"] == args.fold]
        print(f"Tuning on fold {args.fold}: {len(val_df)} samples")

        with open(args.vocab_path, "r") as f:
            vocabs = json.load(f)

        best = tune_on_validation_fold(
            model, processor, val_df, vocabs, args.kenlm_path, config.BeamSearchConfig()
        )
        print(f"\nBest hyperparameters: α={best[0]}, β={best[1]}")


if __name__ == "__main__":
    main()
