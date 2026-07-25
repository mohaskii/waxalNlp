"""
Download WAXAL Audio from Hugging Face — Selective Download
============================================================

Reads Train.csv / Test.csv and downloads ONLY the required audio files
from google/WaxalNLP on Hugging Face, using streaming to avoid downloading
the full 1.06 TB dataset.

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

from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf
import numpy as np

import config

# ---------------------------------------------------------------------------
# Config — matches competition data
# ---------------------------------------------------------------------------
DATASET_ID = config.HF_DATASET_ID
LANG_TO_CONFIG = config.LANG_TO_HF_CONFIG

HF_SPLITS = {
    "train": ["train", "validation"],   # these go to data/Train/
    "test": ["test"],                    # these go to data/Test/
}

# Competition CSV columns
CSV_TRAIN_ID_COL = "id"
CSV_TRAIN_LANG_COL = "language"
CSV_TRAIN_SPLIT_COL = "original_split"
CSV_TEST_ID_COL = "ID"


def _read_competition_ids(data_dir: str) -> dict[str, set[str]]:
    """
    Read Train.csv and Test.csv, return the set of required audio IDs
    for each target directory (Train / Test).

    Returns:
        {"Train": {"lug_96123", "lin_12345", ...},
         "Test":  {"lug_96114", ...}}
    """
    required: dict[str, set[str]] = {"Train": set(), "Test": set()}

    # --- Train.csv ---
    train_path = os.path.join(data_dir, "Train.csv")
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(CSV_TRAIN_ID_COL, "").strip()
                if rid:
                    required["Train"].add(rid)
    else:
        print(f"WARNING: {train_path} not found — no training IDs loaded.")

    # --- Test.csv ---
    test_path = os.path.join(data_dir, "Test.csv")
    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(CSV_TEST_ID_COL, "").strip()
                if rid:
                    required["Test"].add(rid)
    else:
        print(f"WARNING: {test_path} not found — no test IDs loaded.")

    # Sanity check
    overlap = required["Train"] & required["Test"]
    if overlap:
        print(f"WARNING: {len(overlap)} IDs appear in both Train and Test CSVs!")

    total = len(required["Train"]) + len(required["Test"])
    print(f"Required audio files: {len(required['Train'])} train + "
          f"{len(required['Test'])} test = {total} total")
    return required


def _guess_competition_id(example: dict, lang_code: str) -> str | None:
    """
    Try to reconstruct the competition ID (e.g. 'lug_96123') from an
    HF dataset sample by checking likely ID fields.

    The competition IDs follow the pattern: {lang}_{number}
    e.g. lug_96123, lin_4521, sna_7890

    This function tries multiple field names that the HF dataset might use.
    """
    candidates = ["id", "key", "index", "audio_id", "utt_id", "utterance_id"]
    for field in candidates:
        val = example.get(field)
        if val is not None:
            # The HF value might already be the full ID, or just the number part
            sval = str(val).strip()
            if sval.startswith(f"{lang_code}_"):
                return sval   # already has prefix: "lug_96123"
            if sval.isdigit():
                return f"{lang_code}_{sval}"  # just number: "96123"
            # Might be something else — return as-is
            return f"{lang_code}_{sval}"

    # Fallback: no ID field found at all
    return None


def _resolve_id_field(example: dict) -> str:
    """
    Best-effort: return the most likely ID field name from a sample.
    Useful for --probe so the user knows what the dataset looks like.
    """
    for field in ["id", "key", "index", "audio_id", "utt_id", "utterance_id"]:
        if field in example:
            return field
    return "<unknown — no ID field found>"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def probe():
    """Inspect the HF dataset structure for one language."""
    lang = "lin"
    config_name = LANG_TO_CONFIG[lang]

    print(f"\n{'='*60}")
    print(f"Probing: {DATASET_ID} / {config_name} (split: train)")
    print(f"{'='*60}")

    ds = load_dataset(DATASET_ID, name=config_name, split="train", streaming=True)
    sample = next(iter(ds))

    print(f"\n📋 Sample keys and types:")
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"  • {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, np.ndarray):
            print(f"  • {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  • {k}: list, len={len(v)}")
        else:
            print(f"  • {k}: {type(v).__name__} = {str(v)[:100]}")

    # ID field
    id_field = _resolve_id_field(sample)
    print(f"\n🔑 Likely ID field: '{id_field}'")

    if id_field in sample:
        raw_id = sample[id_field]
        guessed = _guess_competition_id(sample, lang)
        print(f"    Raw value:      {raw_id}")
        print(f"    Competition ID: {guessed}")

    print(f"\n💡 Check if this ID format matches your Train.csv / Test.csv IDs.")
    print(f"   If the guessed ID doesn't match, the script won't download correctly.\n")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(target_dir: str = "./data", lang: str | None = None):
    """
    Main download routine.

    For each language (+ optionally filtered), iterate through the HF
    dataset splits in streaming mode, match against the required IDs
    from the competition CSVs, and save matching audio as WAV files.
    """
    # 1. Read required IDs from competition CSVs
    required = _read_competition_ids(target_dir)
    total_required = len(required["Train"]) + len(required["Test"])
    if total_required == 0:
        print("ERROR: No IDs found in Train.csv / Test.csv.")
        print(f"Make sure both CSVs exist in: {target_dir}/")
        sys.exit(1)

    # 2. Determine which languages to process
    langs = [lang] if lang else list(LANG_TO_CONFIG.keys())

    # 3. For each language & required split, stream + save
    downloaded_count = 0
    skipped_count = 0

    hf_token = os.environ.get("HF_TOKEN", None)

    for lang_code in langs:
        config_name = LANG_TO_CONFIG[lang_code]

        for target_subdir, hf_split_names in HF_SPLITS.items():
            output_dir = os.path.join(target_dir, target_subdir)
            os.makedirs(output_dir, exist_ok=True)

            for hf_split in hf_split_names:
                print(f"\n📥 Streaming: {lang_code} / {hf_split} → {output_dir}/")

                ds = load_dataset(
                    DATASET_ID,
                    name=config_name,
                    split=hf_split,
                    streaming=True,
                    token=hf_token,
                )

                # Cast audio to target sample rate
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))

                # Count how many samples from this split are needed
                ids_to_find: set[str] = set()
                if target_subdir == "Train":
                    # For training: keep all IDs from this language
                    ids_to_find = {
                        rid for rid in required["Train"]
                        if rid.startswith(f"{lang_code}_")
                    }
                else:
                    ids_to_find = {
                        rid for rid in required["Test"]
                        if rid.startswith(f"{lang_code}_")
                    }

                relevant_count = len(ids_to_find)
                if relevant_count == 0:
                    print(f"    No required IDs for {lang_code}/{hf_split} — skipping.")
                    continue

                pbar = tqdm(
                    total=relevant_count,
                    desc=f"  {lang_code}/{hf_split}",
                    unit="files",
                )

                found_in_split = 0
                for example in ds:
                    comp_id = _guess_competition_id(example, lang_code)
                    if comp_id is None or comp_id not in ids_to_find:
                        continue

                    # This sample is needed — save it
                    output_path = os.path.join(output_dir, f"{comp_id}.wav")

                    if os.path.exists(output_path):
                        skipped_count += 1
                    else:
                        audio_data = example["audio"]
                        sf.write(
                            output_path,
                            audio_data["array"],
                            audio_data["sampling_rate"],
                        )
                        downloaded_count += 1

                    found_in_split += 1
                    pbar.update(1)
                    pbar.set_postfix(downloaded=downloaded_count)

                    if found_in_split >= relevant_count:
                        break  # all required IDs for this split found

                pbar.close()

                if found_in_split < relevant_count:
                    print(
                        f"  ⚠️  Found {found_in_split}/{relevant_count} required IDs "
                        f"for {lang_code}/{hf_split}. "
                        f"Missing: {relevant_count - found_in_split}"
                    )
                else:
                    print(f"  ✅ All {relevant_count} files for {lang_code}/{hf_split} found.")

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded_count} new files")
    print(f"  Skipped (already exist): {skipped_count} files")

    # Count final files
    for subdir in ["Train", "Test"]:
        dir_path = os.path.join(target_dir, subdir)
        if os.path.exists(dir_path):
            n_files = len([f for f in os.listdir(dir_path) if f.endswith(".wav")])
            print(f"  {subdir}/: {n_files} audio files")
    print(f"{'='*60}\n")


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
        "--data-dir",
        default=config.DATA_DIR,
        help=f"Directory containing Train.csv / Test.csv (default: {config.DATA_DIR})",
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
        download(target_dir=args.data_dir, lang=args.lang)


if __name__ == "__main__":
    main()
