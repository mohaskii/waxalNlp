"""
Download WAXAL Audio from Hugging Face — Selective Download
============================================================

Reads Train.csv / Test.csv and downloads ONLY the required audio files
from google/WaxalNLP on Hugging Face, using streaming to avoid downloading
the full 1.06 TB dataset.

Optimizations (v2):
  • Uses Train.csv's 'original_split' column to skip irrelevant HF splits
  • Direct example["id"] lookup (O(1) set check) — no string building
  • Flat global ID set — single check instead of per-language per-split filtering
  • Skips the Audio cast_column (HF native rate is already 16 kHz)
  • Single-pass streaming per HF split

Usage:
    # Step 1 — Probe the dataset structure (shows field names + ID format):
    python download_data.py --probe

    # Step 2 — Download audio for all required IDs:
    python download_data.py --download

    # Step 2b — Download for a single language only (resume-friendly):
    python download_data.py --download --lang lin

Notes:
    - Run on Kaggle or any machine with the `datasets` library installed.
    - Hugging Face token recommended for faster downloads (set HF_TOKEN env var).
    - The script skips already-downloaded files, so it's safe to interrupt and resume.
"""

import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Config — matches competition data
# ---------------------------------------------------------------------------
DATASET_ID = config.HF_DATASET_ID
LANG_TO_CONFIG = config.LANG_TO_HF_CONFIG

# Competition CSV columns
CSV_TRAIN_ID_COL = "id"
CSV_TRAIN_LANG_COL = "language"
CSV_TRAIN_SPLIT_COL = "original_split"
CSV_TEST_ID_COL = "ID"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _authenticate() -> str | None:
    """Set up HF authentication token for this session.

    Sets the token as an environment variable (which the datasets library
    picks up automatically) and also returns it for explicit passing.

    Unlike huggingface_hub.login(), this does NOT call whoami — so it works
    even with read-only tokens that can't pass the whoami check but CAN
    download public datasets.

    Returns:
        The token string, or None if no token is configured.
    """
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("✅ HF token configured (read-only — no whoami check needed)")
    else:
        print("⚠️  No HF_TOKEN set — downloads may be rate-limited.")
    return token


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------
def _read_competition_ids(csv_dir: str) -> tuple[dict[str, set[str]], set[str]]:
    """
    Read Train.csv and Test.csv.

    Train.csv columns: id, transcription, language, original_split
    Test.csv  columns: ID

    We use Train.csv's 'original_split' column to know exactly which HF
    split each sample lives in — this lets us skip irrelevant splits
    during download instead of scanning the full HF dataset.

    Args:
        csv_dir: Directory containing Train.csv and Test.csv

    Returns:
        split_map:  {hf_split: set_of_ids}  — which IDs belong to which HF split
        all_ids:    set of ALL required IDs for O(1) lookup during streaming
    """
    split_map: dict[str, set[str]] = {}
    all_ids: set[str] = set()

    # --- Train.csv (has original_split column) ---
    train_path = os.path.join(csv_dir, "Train.csv")
    train_count = 0
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(CSV_TRAIN_ID_COL, "").strip()
                split = row.get(CSV_TRAIN_SPLIT_COL, "").strip()
                if rid and split in ("train", "validation", "test"):
                    split_map.setdefault(split, set()).add(rid)
                    all_ids.add(rid)
                    train_count += 1
    else:
        print(f"WARNING: {train_path} not found — no training IDs loaded.")

    # --- Test.csv (no original_split → all go to HF 'test') ---
    test_path = os.path.join(csv_dir, "Test.csv")
    test_count = 0
    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(CSV_TEST_ID_COL, "").strip()
                if rid:
                    split_map.setdefault("test", set()).add(rid)
                    all_ids.add(rid)
                    test_count += 1
    else:
        print(f"WARNING: {test_path} not found — no test IDs loaded.")

    print(
        f"Required audio files: {train_count} from Train.csv + "
        f"{test_count} from Test.csv = {len(all_ids)} total"
    )
    for split_name in sorted(split_map.keys()):
        print(f"  HF split '{split_name}': {len(split_map[split_name])} files needed")

    return split_map, all_ids


# ---------------------------------------------------------------------------
# ID helpers (used by --probe only)
# ---------------------------------------------------------------------------
def _guess_competition_id(example: dict, lang_code: str) -> str | None:
    """Try to reconstruct the competition ID from an HF dataset sample."""
    candidates = ["id", "key", "index", "audio_id", "utt_id", "utterance_id"]
    for field in candidates:
        val = example.get(field)
        if val is not None:
            sval = str(val).strip()
            if sval.startswith(f"{lang_code}_"):
                return sval
            if sval.isdigit():
                return f"{lang_code}_{sval}"
            return f"{lang_code}_{sval}"
    return None


def _resolve_id_field(example: dict) -> str:
    """Return the most likely ID field name from a sample."""
    for field in ["id", "key", "index", "audio_id", "utt_id", "utterance_id"]:
        if field in example:
            return field
    return "<unknown — no ID field found>"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def probe():
    """Inspect the HF dataset structure for one language."""
    token = _authenticate()
    lang = "lin"
    config_name = LANG_TO_CONFIG[lang]

    print(f"\n{'=' * 60}")
    print(f"Probing: {DATASET_ID} / {config_name} (split: train)")
    print(f"{'=' * 60}")

    ds = load_dataset(
        DATASET_ID, name=config_name, split="train", streaming=True, token=token
    )
    sample = next(iter(ds))

    print("\n📋 Sample keys and types:")
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"  • {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, np.ndarray):
            print(f"  • {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  • {k}: list, len={len(v)}")
        else:
            print(f"  • {k}: {type(v).__name__} = {str(v)[:100]}")

    id_field = _resolve_id_field(sample)
    print(f"\n🔑 Likely ID field: '{id_field}'")

    if id_field in sample:
        raw_id = sample[id_field]
        guessed = _guess_competition_id(sample, lang)
        print(f"    Raw value:      {raw_id}")
        print(f"    Competition ID: {guessed}")

    print("\n💡 Check if this ID format matches your Train.csv / Test.csv IDs.")
    print("   If the guessed ID doesn't match, the script won't download correctly.\n")


# ---------------------------------------------------------------------------
# Download (optimized)
# ---------------------------------------------------------------------------
def download(audio_dir: str = config.DATA_DIR, csv_dir: str = "./data", lang: str | None = None):
    """
    Main download routine — optimized v2.

    Args:
        audio_dir: Where to save audio files (e.g. /tmp/data on Kaggle)
        csv_dir:   Where to find Train.csv / Test.csv (e.g. ./data)
        lang:      Optional single language to download

    Strategy:
    1. Parse CSVs once → split_map {hf_split: set_of_ids} + flat all_ids set
    2. For each HF split that HAS required IDs, open it using streaming,
       filtered to the target language
    3. Stream samples, check example["id"] against the relevant set (O(1))
    4. Break early when all required IDs for a split are found

    Key wins over v1:
    • Uses original_split to skip irrelevant HF splits (no wasted streaming)
    • Direct example["id"] access — no string ops per sample
    • Single flat set — one __contains__ call per sample (was: per-lang per-split per-sample)
    • No Audio.cast_column overhead
    """
    token = _authenticate()

    # 1. Parse CSVs → split_map tells us which HF splits have required IDs
    split_map, all_ids = _read_competition_ids(csv_dir)
    if not all_ids:
        print("ERROR: No IDs found in Train.csv / Test.csv.")
        print(f"Make sure both CSVs exist in: {target_dir}/")
        sys.exit(1)

    # 2. Determine which languages to process
    langs = [lang] if lang else list(LANG_TO_CONFIG.keys())
    print(f"\nLanguages to download: {', '.join(langs)}")

    downloaded_count = 0
    skipped_count = 0

    # Map HF split → local output subdirectory
    HF_SPLIT_TO_OUTPUT = {
        "train": "Train",
        "validation": "Train",
        "test": "Test",
    }

    for lang_code in langs:
        config_name = LANG_TO_CONFIG[lang_code]

        for hf_split, output_subdir in HF_SPLIT_TO_OUTPUT.items():
            # Skip splits with zero required IDs (saves loading them at all)
            ids_in_split = split_map.get(hf_split)
            if not ids_in_split:
                continue

            # Filter to this language only
            prefix = f"{lang_code}_"
            relevant_ids = {rid for rid in ids_in_split if rid.startswith(prefix)}
            if not relevant_ids:
                continue

            output_dir = os.path.join(audio_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)

            print(
                f"\n📥 {lang_code}/{hf_split} → {output_dir}/ "
                f"(need {len(relevant_ids)}/{len(ids_in_split)} IDs in this split)"
            )

            ds = load_dataset(
                DATASET_ID,
                name=config_name,
                split=hf_split,
                streaming=True,
                token=token,
            )

            pbar = tqdm(
                total=len(relevant_ids),
                desc=f"  {lang_code}/{hf_split}",
                unit="files",
            )

            found_ids: set[str] = set()
            for example in ds:
                rid = example.get("id")
                # O(1) set lookup, no string building
                if rid is None or rid not in relevant_ids:
                    continue

                output_path = os.path.join(output_dir, f"{rid}.wav")

                if os.path.exists(output_path):
                    skipped_count += 1
                else:
                    audio = example["audio"]
                    sf.write(output_path, audio["array"], audio["sampling_rate"])
                    downloaded_count += 1

                found_ids.add(rid)
                pbar.update(1)
                pbar.set_postfix(downloaded=downloaded_count)

                # All required IDs for this lang/split found → stop scanning
                if found_ids >= relevant_ids:
                    break

            pbar.close()
            missing = len(relevant_ids) - len(found_ids)
            if missing > 0:
                print(
                    f"  ⚠️  Missing {missing} file(s) for {lang_code}/{hf_split}"
                )
            else:
                print(
                    f"  ✅ All {len(relevant_ids)} files for {lang_code}/{hf_split} found"
                )

    # 4. Summary
    print(f"\n{'=' * 60}")
    print("Download complete!")
    print(f"  Downloaded: {downloaded_count} new files")
    print(f"  Skipped (already exist): {skipped_count} files")

    for subdir in ["Train", "Test"]:
        dir_path = os.path.join(target_dir, subdir)
        if os.path.exists(dir_path):
            n_files = len([f for f in os.listdir(dir_path) if f.endswith(".wav")])
            print(f"  {subdir}/: {n_files} audio files")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Selectively download WAXAL audio from Hugging Face"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe the HF dataset structure (no download)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download audio files matching Train.csv / Test.csv IDs",
    )
    parser.add_argument(
        "--audio-dir",
        default=config.DATA_DIR,
        help=f"Where to save downloaded audio (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--csv-dir",
        default="./data",
        help=f"Directory containing Train.csv / Test.csv (default: ./data)",
    )
    parser.add_argument(
        "--lang",
        choices=list(LANG_TO_CONFIG.keys()),
        default=None,
        help="Download only this language (default: all three)",
    )

    args = parser.parse_args()

    if not args.probe and not args.download:
        parser.print_help()
        print("\n❌ Specify at least one of: --probe or --download")
        sys.exit(1)

    if args.probe:
        probe()

    if args.download:
        download(audio_dir=args.audio_dir, csv_dir=args.csv_dir, lang=args.lang)


if __name__ == "__main__":
    main()
