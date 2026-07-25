"""
Step 1: Data Preprocessing & Validation Strategy
================================================
- Speaker-disjoint GroupKFold (5 folds)
- Unicode NFC normalization of transcripts
- Build target vocabularies per language
- Save processed artifacts for downstream steps
"""

import csv
import json
import os

import pandas as pd
from sklearn.model_selection import GroupKFold

import config
from utils import build_vocab, normalize_text


def load_data():
    """Load Train.csv and Test.csv. Return dataframes.

    Uses csv.DictReader to safely handle commas & quotes inside transcriptions
    that trip up pd.read_csv's C engine.

    Loads speaker_map.json (saved by download_data.py) for the Speaker_ID
    column needed by GroupKFold.
    """
    # --- Load speaker map (id -> speaker_id) from download step ---
    speaker_map: dict[str, str] = {}
    speaker_path = os.path.join(config.DATA_DIR, "speaker_map.json")
    if os.path.exists(speaker_path):
        with open(speaker_path, "r") as f:
            speaker_map = json.load(f)
        print(f"Loaded {len(speaker_map)} speaker IDs from {speaker_path}")
    else:
        print(f"WARNING: {speaker_path} not found - using language prefix as fallback.")
        print("GroupKFold will group by language, not by speaker.")

    # --- Train.csv ---
    train_rows = []
    with open(config.TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id", "").strip()
            if not rid:
                continue  # skip malformed rows
            train_rows.append({
                "Audio_ID": rid,
                "Speaker_ID": speaker_map.get(rid, rid.split("_", 1)[0]),
                "Transcript": row.get("transcription", ""),
                "Language": row.get("language", "").strip(),
                "original_split": row.get("original_split", "").strip(),
            })
    train_df = pd.DataFrame(train_rows)

    # --- Test.csv ---
    test_rows = []
    test_path = config.TEST_CSV
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("ID", "").strip()
                if rid:
                    test_rows.append({"Audio_ID": rid})
    test_df = pd.DataFrame(test_rows)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Languages:     {sorted(train_df['Language'].unique())}")
    return train_df, test_df


def validate_columns(train_df: pd.DataFrame) -> None:
    """Ensure required columns are present."""
    required = {"Audio_ID", "Speaker_ID", "Transcript", "Language"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns in Train.csv: {missing}")
    print(f"Columns OK: {sorted(train_df.columns.tolist())}")


def create_folds(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5 speaker-disjoint folds using GroupKFold.
    This matches the Phase 2 private test split (unseen speakers).
    """
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    train_df = train_df.copy()
    train_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(train_df, groups=train_df["Speaker_ID"])
    ):
        train_df.loc[train_df.index[val_idx], "fold"] = fold_idx

    # Print fold stats
    for f in range(config.N_FOLDS):
        fold_mask = train_df["fold"] == f
        n_speakers = train_df.loc[fold_mask, "Speaker_ID"].nunique()
        n_samples = fold_mask.sum()
        print(f"  Fold {f}: {n_samples} samples, {n_speakers} speakers")
    return train_df


def normalize_transcripts(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Apply NFC normalization to all transcripts.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["Transcript_normalized"] = train_df["Transcript"].apply(normalize_text)
    if "Transcript" in test_df.columns:
        test_df["Transcript_normalized"] = test_df["Transcript"].apply(normalize_text)
    else:
        test_df["Transcript_normalized"] = ""

    return train_df, test_df


def build_language_vocabs(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Build per-language character vocabularies from normalized transcripts.
    Also build a combined vocab across all languages.
    """
    vocabs = {}
    for lang in config.LANGUAGES:
        lang_df = train_df[train_df["Language"] == lang]
        transcripts = lang_df["Transcript_normalized"].tolist()
        vocabs[lang] = build_vocab(transcripts)
        n_tok = len(vocabs[lang])
        print(f"  {lang}: {n_tok} tokens (chars: {n_tok - 4})")

    # Combined vocabulary
    all_transcripts = train_df["Transcript_normalized"].tolist()
    vocabs["combined"] = build_vocab(all_transcripts)
    n_tok = len(vocabs['combined'])
    print(f"  Combined: {n_tok} tokens (chars: {n_tok - 4})")

    return vocabs


def main():
    print("=" * 60)
    print("STEP 1: Data Preprocessing & Cross-Validation Setup")
    print("=" * 60)

    # 1. Load data
    train_df, test_df = load_data()
    validate_columns(train_df)

    # 2. Create speaker-disjoint folds
    print("\n--- Creating Speaker-Disjoint Folds ---")
    train_df = create_folds(train_df)

    # 3. Normalize transcripts
    print("\n--- Normalizing Transcripts (NFC) ---")
    train_df, test_df = normalize_transcripts(train_df, test_df)
    sample = train_df["Transcript_normalized"].iloc[0]
    print(f"  Sample normalized: {sample[:80]}...")

    # 4. Build vocabularies
    print("\n--- Building Character Vocabularies ---")
    vocabs = build_language_vocabs(train_df)

    # 5. Save processed artifacts
    print("\n--- Saving Artifacts ---")
    train_df.to_csv(os.path.join(config.OUTPUT_DIR, "train_folds.csv"), index=False)
    test_df.to_csv(os.path.join(config.OUTPUT_DIR, "test_normalized.csv"), index=False)

    with open(os.path.join(config.OUTPUT_DIR, "vocabs.json"), "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)

    # Also save combined transcripts for KenLM training
    all_text_path = os.path.join(config.OUTPUT_DIR, "all_transcripts.txt")
    with open(all_text_path, "w", encoding="utf-8") as f:
        f.writelines(t + "\n" for t in train_df["Transcript_normalized"])

    print(f"\nArtifacts saved to {config.OUTPUT_DIR}/")
    print("  - train_folds.csv")
    print("  - test_normalized.csv")
    print("  - vocabs.json")
    print("  - all_transcripts.txt")
    print("\nStep 1 complete!")


if __name__ == "__main__":
    main()
