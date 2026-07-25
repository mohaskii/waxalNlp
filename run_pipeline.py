#!/usr/bin/env python3
"""
Master Pipeline Runner for WAXAL ASR Challenge
==============================================
Orchestrates the full workflow:
  0. Download Data  →  download_data.py   (optional, needs internet)
  1. Preprocessing  →  step1_preprocessing.py
  2. MMS-1B Train   →  step2_train_mms.py
  3. KenLM Train    →  step3_train_kenlm.py
  4. w2v-BERT Train  →  step4_train_w2vbert.py
  5. Ensemble        →  step5_ensemble.py

Usage:
  # Full pipeline (run on Kaggle with internet):
  python run_pipeline.py --steps 0,1,2,3,4,5

  # Selective runs:
  python run_pipeline.py --steps 1,3        # preprocess + LM only
  python run_pipeline.py --steps 5          # ensemble only (requires trained models)
  python run_pipeline.py --steps 0,1        # download data + preprocess
"""

import argparse
import subprocess
import sys

STEPS: dict[int, tuple[str, str, list[int]]] = {
    0: ("Data Download (HF)", "download_data.py", []),
    1: ("Preprocessing & CV", "step1_preprocessing.py", []),
    2: ("MMS-1B Fine-Tuning", "step2_train_mms.py", [1]),
    3: ("KenLM Language Model", "step3_train_kenlm.py", [1]),
    4: ("w2v-BERT 2.0 Fine-Tuning", "step4_train_w2vbert.py", [1]),
    5: ("Ensemble & Submission", "step5_ensemble.py", [2, 3, 4]),
}


def run_step(step_num: int, extra_args: list[str] | None = None) -> bool:
    name, script, _deps = STEPS[step_num]
    print(f"\n{'#' * 60}")
    print(f"# STEP {step_num}: {name}")
    print(f"{'#' * 60}")

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    ret = subprocess.run(cmd, check=False)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="WAXAL ASR Pipeline Runner")
    _ = parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated steps to run (e.g., '1,3,5')",
    )
    _ = parser.add_argument(
        "--step_args",
        type=str,
        default="",
        help=(
            "Extra arguments to pass to each step (semicolon-separated per step, "
            "e.g., ';--fold 0;--alpha 2.0 --beta 0.5')"
        ),
    )
    args = parser.parse_args()

    step_list = [int(s.strip()) for s in args.steps.split(",")]
    step_args_list = (
        args.step_args.split(";") if args.step_args else [""] * len(step_list)
    )

    for i, step_num in enumerate(step_list):
        step_extra = (
            step_args_list[i].strip().split() if i < len(step_args_list) else []
        )
        ok = run_step(step_num, step_extra if step_extra else None)
        if not ok:
            print(f"\nERROR: Step {step_num} failed. Aborting pipeline.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("Submission files are in ./output/submissions/")


if __name__ == "__main__":
    main()
