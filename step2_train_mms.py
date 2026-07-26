"""
Step 2: Fine-Tune Meta MMS-1B with LoRA
=======================================
- Backbone: facebook/mms-1b-all
- PEFT/LoRA adapter training (not full 1B parameters)
- SpecAugment for generalization
- fp16 + gradient accumulation for Kaggle GPU
- Push checkpoints to Hugging Face Hub
"""

import argparse
import gc
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)

import config
from utils import (
    DataCollatorCTCWithPadding,
    compute_metrics,
    load_audio,
    print_gpu_info,
    resolve_audio_path,
)


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------
def prepare_dataset_for_fold(
    train_df: pd.DataFrame, fold: int, processor, tokenizer
) -> DatasetDict:
    """
    Build a Hugging Face DatasetDict for a single fold.
    Uses speaker-disjoint split: all data NOT in `fold` is training;
    data IN `fold` is validation.
    """
    # Split
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    def _make_hf_dataset(df: pd.DataFrame) -> Dataset:
        records = []
        for _, row in df.iterrows():
            audio_path = os.path.join(config.TRAIN_AUDIO_DIR, str(row["Audio_ID"]))
            try:
                audio_path = resolve_audio_path(audio_path)
            except FileNotFoundError:
                print(f"WARNING: {audio_path} not found, skipping")
                continue
            # Handle NaN transcripts (empty rows — skip them)
            transcript = row["Transcript_normalized"]
            if isinstance(transcript, float):
                continue  # NaN — skip this sample entirely
            records.append(
                {
                    "audio_path": audio_path,
                    "transcript": transcript,
                }
            )
        return Dataset.from_list(records)

    def _preprocess(batch):
        # Load and extract features
        audio_arrays = [load_audio(p) for p in batch["audio_path"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=config.SAMPLING_RATE,
            return_tensors=None,  # return lists
            padding=False,  # we pad in the collator
        )
        batch["input_values"] = inputs["input_values"]
        # Tokenize labels
        labels = tokenizer(
            batch["transcript"],
            padding=False,
            truncation=False,
        )["input_ids"]
        batch["labels"] = labels
        return batch

    # We need the raw feature extractor (not the combined processor) for audio
    if isinstance(processor, Wav2Vec2Processor):
        feature_extractor = processor.feature_extractor
        tokenizer_part = processor.tokenizer
    else:
        feature_extractor = processor
        tokenizer_part = tokenizer

    ds_train = _make_hf_dataset(train_df.loc[train_mask])
    ds_val = _make_hf_dataset(train_df.loc[val_mask])

    ds_train = ds_train.map(
        _preprocess, batched=True, batch_size=32,
        remove_columns=["audio_path", "transcript"],
    )
    ds_val = ds_val.map(
        _preprocess, batched=True, batch_size=32,
        remove_columns=["audio_path", "transcript"],
    )

    return (
        DatasetDict({"train": ds_train, "validation": ds_val}),
        feature_extractor,
        tokenizer_part,
    )


# ---------------------------------------------------------------------------
# LoRA Setup
# ---------------------------------------------------------------------------
def apply_lora(model, cfg: config.MMSConfig):
    """Wrap a Wav2Vec2ForCTC model with LoRA adapters."""
    # Kaggle ships torchao 0.10 but peft needs 0.16+
    try:
        import torchao
        from packaging.version import parse as _v
        if _v(torchao.__version__) < _v("0.16.0"):
            raise ImportError
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16.0"])

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Compute Metrics Function for Trainer
# ---------------------------------------------------------------------------
def make_compute_metrics(tokenizer):
    """Factory that returns a compute_metrics function using an external WER/CER lib."""

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)

        # Decode
        pred_strs = tokenizer.batch_decode(pred_ids)
        # Replace -100 (CTC padding) with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_strs = tokenizer.batch_decode(labels)

        # Clean up special tokens
        pred_strs = [s.replace(tokenizer.pad_token, "").strip() for s in pred_strs]
        label_strs = [s.replace(tokenizer.pad_token, "").strip() for s in label_strs]

        return compute_metrics(pred_strs, label_strs)

    return _compute_metrics


# ---------------------------------------------------------------------------
# Main Training Loop (All Folds)
# ---------------------------------------------------------------------------
def train_single_fold(fold: int, train_df: pd.DataFrame, cfg: config.MMSConfig):
    print(f"\n{'=' * 60}")
    print(f"TRAINING FOLD {fold + 1}/{config.N_FOLDS}")
    print(f"{'=' * 60}")

    # 1. Load processor & model
    print("Loading MMS-1B model...")
    processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCTC.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
    )
    tokenizer = processor.tokenizer  # MMS uses Wav2Vec2Processor with a tokenizer

    # 2. Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model, cfg)

    # 3. Prepare dataset
    print(f"Preparing fold {fold} data...")
    datasets, feature_extractor, tokenizer_part = prepare_dataset_for_fold(
        train_df, fold, processor, tokenizer
    )
    print(f"  Train size: {len(datasets['train'])}")
    print(f"  Val size:   {len(datasets['validation'])}")

    # 4. Data collator
    data_collator = DataCollatorCTCWithPadding(feature_extractor, tokenizer_part)

    # 5. Training args
    output_dir = os.path.join(config.CHECKPOINT_DIR, f"mms_fold_{fold}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_train_epochs,
        fp16=cfg.fp16,
        evaluation_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="combined",
        greater_is_better=False,
        push_to_hub=False,  # we push manually after training (final adapter only)
        hub_model_id=f"{config.HF_USERNAME}/{cfg.hub_model_id}-fold{fold}",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer_part,
        compute_metrics=make_compute_metrics(tokenizer_part),
    )

    # 7. Train
    if hasattr(model.config, "apply_spec_augment"):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = cfg.specaug_time_mask
        model.config.mask_feature_prob = 0.0
        model.config.mask_feature_length = cfg.specaug_freq_mask

    print("Starting training...")
    trainer.train()

    # 8. Save final adapter
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # 9. Push final adapter to HF Hub (safety — session may time out)
    if cfg.push_to_hub:
        try:
            hub_repo = f"{config.HF_USERNAME}/{cfg.hub_model_id}-fold{fold}"
            print(f"Pushing adapter to HF Hub: {hub_repo} ...")
            model.push_to_hub(hub_repo, token=config.HF_TOKEN)
            print(f"✅ Pushed to https://huggingface.co/{hub_repo}")
        except (OSError, ValueError, RuntimeError) as e:
            print(f"⚠️  HF Hub push failed: {e}")
            print(f"   Adapter saved locally at {adapter_path} — push manually later.")

    # Cleanup
    del model, trainer, datasets
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only this fold (0-4). Default: all 5 folds.")
    args = parser.parse_args()

    print_gpu_info()

    train_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "train_folds.csv"))
    cfg = config.MMSConfig()

    print(f"Model:      {cfg.model_name}")
    print(f"LoRA r:     {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"Batch size: {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps} accum")
    print(f"Epochs:     {cfg.num_train_epochs}")
    print(f"FP16:       {cfg.fp16}")

    folds = [args.fold] if args.fold is not None else range(config.N_FOLDS)
    for fold in folds:
        train_single_fold(fold, train_df, cfg)

    print("\nAll folds trained! Checkpoints in:", config.CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
