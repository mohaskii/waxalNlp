"""
Step 3: Build & Integrate KenLM Language Model
==============================================
- Collect training transcripts + optional external text
- Train a 5-gram KenLM model
- Tune alpha/beta hyperparameters using pyctcdecode CTC beam search
"""

import itertools
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import torch
from pyctcdecode.decoder import build_ctcdecoder

import config
from utils import compute_metrics, load_audio, normalize_text


def _ensure_kenlm_binaries():
    """Build lmplz and build_binary from source if they don't exist.

    pip install kenlm only gives Python bindings, not the CLI tools.
    This builds the C++ binaries (~2 min on Kaggle).
    """
    if shutil.which("lmplz") and shutil.which("build_binary"):
        return

    print("Building kenlm binaries from source (one-time, ~2 min)...")
    subprocess.run(
        "apt-get update -qq && "
        "apt-get install -y -qq cmake build-essential libeigen3-dev "
        "libboost-all-dev && "
        "cd /tmp && rm -rf kenlm && git clone --depth 1 https://github.com/kpu/kenlm.git && "
        "cd kenlm && mkdir -p build && cd build && "
        "cmake .. -DKENLM_MAX_ORDER=8 && make -j$(nproc) && "
        "cp bin/lmplz bin/build_binary /usr/local/bin/",
        shell=True, check=True,
    )

    if not shutil.which("lmplz"):
        raise RuntimeError(
            "lmplz build failed. Run manually:\n"
            "  apt-get install -y cmake build-essential libboost-all-dev\n"
            "  cd /tmp && git clone --depth 1 https://github.com/kpu/kenlm.git\n"
            "  cd /tmp/kenlm && mkdir -p build && cd build\n"
            "  cmake .. -DKENLM_MAX_ORDER=8 && make -j$(nproc)\n"
            "  cp bin/lmplz bin/build_binary /usr/local/bin/"
        )
    print("kenlm binaries ready.")


# ---------------------------------------------------------------------------
# 1. Prepare LM training corpus
# ---------------------------------------------------------------------------
def prepare_corpus() -> str:
    """
    Collect all normalized transcripts + any external text files.
    Returns path to the combined corpus text file.
    """
    corpus_path = os.path.join(config.KENLM_DIR, "lm_corpus.txt")

    # Load normalized training transcripts    # drop NaN (empty rows) + ensure strings
    train_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "train_folds.csv"))
    lines = [str(t) for t in train_df["Transcript_normalized"].dropna().tolist()]

    # Optional: add external text (Wikipedia, Leipzig dumps)
    external_dir = os.path.join(config.DATA_DIR, "external_text")
    if os.path.isdir(external_dir):
        print(f"Found external text directory: {external_dir}")
        for fname in sorted(os.listdir(external_dir)):
            fpath = os.path.join(external_dir, fname)
            if fname.endswith(".txt"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = normalize_text(line)
                        if line:
                            _ = lines.append(line)
                print(f"  Loaded {fname}")

    # Write corpus
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in lines:
            # pandas NaN becomes float; skip empty/NaN rows
            if not isinstance(line, str) or not line.strip():
                continue
            _ = f.write(line + "\n")

    print(f"LM corpus: {len(lines)} lines -> {corpus_path}")
    return corpus_path


def _find_binary(name: str) -> str:
    """Find a kenlm binary (lmplz or build_binary).

    Tries: PATH, common install locations, and Python bin directory.
    Raises FileNotFoundError if not found.
    """
    # 1. Check PATH
    path = shutil.which(name)
    if path:
        return path

    # 2. Check alongside the Python executable (pip --user installs)
    python_bin = os.path.dirname(sys.executable)
    path = os.path.join(python_bin, name)
    if os.path.isfile(path):
        return path

    # 3. Check /usr/local/bin (common on Kaggle/Linux)
    path = os.path.join("/usr/local/bin", name)
    if os.path.isfile(path):
        return path

    raise FileNotFoundError(
        f"{name} not found. Install kenlm with binaries:\n"
        f"  apt-get install -y -qq cmake build-essential libboost-all-dev && "
        f"pip install https://github.com/kpu/kenlm/archive/master.zip"
    )


# ---------------------------------------------------------------------------
# 2. Train KenLM
# ---------------------------------------------------------------------------
def train_kenlm(corpus_path: str, cfg: config.KenLMConfig) -> str:
    """
    Build a KenLM ARPA file using the `lmplz` binary.
    Requires kenlm to be installed: pip install kenlm
    (which includes the lmplz and build_binary commands).
    """
    arpa_path = os.path.join(config.KENLM_DIR, "lm_5gram.arpa")
    binary_path = os.path.join(config.KENLM_DIR, "lm_5gram.binary")

    # Build ARPA model
    lmplz = _find_binary("lmplz")
    build_bin = _find_binary("build_binary")
    prune_str = " ".join(str(p) for p in cfg.prune_values)
    cmd = (
        f"{lmplz} -o {cfg.ngram_order} "
        f"-S 80% "
        f"--prune {prune_str} "
        f"< {corpus_path} > {arpa_path}"
    )
    print(f"Running: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=False, check=False)
    if ret.returncode != 0:
        print("ERROR: lmplz failed. Make sure `kenlm` is installed (pip install kenlm).")
        sys.exit(1)

    # Build binary (faster loading)
    cmd2 = f"{build_bin} {arpa_path} {binary_path}"
    print(f"Running: {cmd2}")
    _ = subprocess.run(cmd2, shell=True, check=True)

    print(f"KenLM model saved to {binary_path}")
    return binary_path


# ---------------------------------------------------------------------------
# 3. Build pyctcdecode decoder with vocabulary
# ---------------------------------------------------------------------------
def build_decoder(vocab: dict[str, int], kenlm_path: str, alpha: float, beta: float):
    """
    Build a CTC beam search decoder using pyctcdecode.
    `vocab` is a dict like {"a": 0, "b": 1, ...} (WITHOUT special tokens).
    The decoder expects labels in the same order as the model's tokenizer.
    """
    # pyctcdecode needs a list of labels
    labels = [""] * len(vocab)
    for char, idx in vocab.items():
        labels[idx] = char

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta,
    )
    return decoder


# ---------------------------------------------------------------------------
# 4. Hyperparameter grid search (Alpha, Beta)
# ---------------------------------------------------------------------------
def tune_alpha_beta(
    model,
    processor,
    val_df: pd.DataFrame,
    vocabs: dict[str, dict[str, int]],
    kenlm_path: str,
    beam_cfg: config.BeamSearchConfig,
) -> tuple[float, float]:
    """
    Grid-search alpha and beta on a validation fold.
    Returns best (alpha, beta) and the corresponding metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Use combined vocabulary for the decoder labels
    # (or language-specific if you want per-language decoders)
    vocab = vocabs["combined"]

    # Gather ground truth
    references = val_df["Transcript_normalized"].tolist()

    best_combined = float("inf")
    best_params = (1.0, 1.0)

    print(f"\nGrid search over alpha {beam_cfg.alpha_range}, beta {beam_cfg.beta_range}")
    results = []

    for alpha, beta in itertools.product(beam_cfg.alpha_range, beam_cfg.beta_range):
        decoder = build_decoder(vocab, kenlm_path, alpha, beta)
        predictions: list[str] = []

        for _, row in val_df.iterrows():
            audio_path = os.path.join(config.TRAIN_AUDIO_DIR, str(row["Audio_ID"]))
            audio = load_audio_for_model(audio_path, processor)
            with torch.no_grad():
                outputs = model(audio["input_values"].to(device))
            logits = outputs.logits[0].cpu().numpy()
            pred_text = decoder.decode(logits)  # pyctcdecode beam search
            predictions.append(normalize_text(pred_text))

        metrics = compute_metrics(predictions, references)
        results.append((alpha, beta, metrics))
        m = metrics  # shorthand
        print(f"  alpha={alpha:.1f} beta={beta:.1f}  WER={m['wer']:.4f} CER={m['cer']:.4f} Combined={m['combined']:.4f}")

        if metrics["combined"] < best_combined:
            best_combined = metrics["combined"]
            best_params = (alpha, beta)

    print(f"\nBest: alpha={best_params[0]}, beta={best_params[1]} (combined={best_combined:.4f})")

    # Save results
    results_path = os.path.join(config.KENLM_DIR, "beam_search_tuning.json")
    with open(results_path, "w") as f:
        json.dump({
            "best": {"alpha": best_params[0], "beta": best_params[1], "metric": best_combined},
            "all": [{"alpha": a, "beta": b, **m} for a, b, m in results],
        }, f, indent=2)

    return best_params


def load_audio_for_model(path: str, processor) -> dict[str, torch.Tensor]:
    """Helper: load audio and run through feature extractor."""
    audio = load_audio(path)
    inputs = processor(
        audio,
        sampling_rate=config.SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 3: KenLM Language Model")
    print("=" * 60)

    _ensure_kenlm_binaries()  # build lmplz/build_binary if missing

    kenlm_cfg = config.KenLMConfig()
    beam_cfg = config.BeamSearchConfig()

    # 1. Prepare corpus
    print("\n--- Preparing LM Corpus ---")
    corpus_path = prepare_corpus()

    # 2. Train KenLM
    print("\n--- Training KenLM ---")
    kenlm_path = train_kenlm(corpus_path, kenlm_cfg)

    # 3. Load vocabulary
    with open(os.path.join(config.OUTPUT_DIR, "vocabs.json"), "r") as f:
        vocabs = json.load(f)

    # 4. Build a decoder and test with greedy baseline (optional)
    print("\n--- Building CTC Decoder ---")
    _decoder = build_decoder(vocabs["combined"], kenlm_path, alpha=1.5, beta=1.0)
    print("Decoder built successfully!")

    # 5. Tuning instructions
    print("\n--- Hyperparameter Tuning ---")
    print("To tune alpha/beta, load your fine-tuned model and run tune_alpha_beta().")
    bc = beam_cfg
    print(f"KenLM binary:    {kenlm_path}")
    print(f"Beam search cfg: alpha_range={bc.alpha_range}, beta_range={bc.beta_range}")

    print("\nStep 3 complete!")
    print("Next: Use step3_decode.py to decode with the tuned KenLM model.")


if __name__ == "__main__":
    main()
