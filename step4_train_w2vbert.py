"""
Step 4: Train Secondary Diversity Model (w2v-BERT 2.0)
======================================================
- Backbone: facebook/w2v-bert-2.0
- Fine-tune with CTC + LoRA on the same preprocessed folds
- Architecture diversity: Conformer blocks vs. MMS-1B pure Transformer
- Same training regimen as step 2 for fair comparison
"""

import gc
import os

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
# Dataset Preparation (same logic as step 2)
# ---------------------------------------------------------------------------
def prepare_dataset_for_fold(
    train_df: pd.DataFrame, fold: int, processor, tokenizer
) -> DatasetDict:
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    if isinstance(processor, Wav2Vec2Processor):
        feature_extractor = processor.feature_extractor
        tokenizer_part = processor.tokenizer
    else:
        feature_extractor = processor
        tokenizer_part = tokenizer

    def _make_hf_dataset(df: pd.DataFrame) -> Dataset:
        records = []
        for _, row in df.iterrows():
            audio_path = os.path.join(config.TRAIN_AUDIO_DIR, str(row["Audio_ID"]))
            try:
                audio_path = resolve_audio_path(audio_path)
            except FileNotFoundError:
                print(f"WARNING: {audio_path} not found, skipping")
                continue
            records.append(
                {
                    "audio_path": audio_path,
                    "transcript": row["Transcript_normalized"],
                }
            )
        return Dataset.from_list(records)

    def _preprocess(batch):
        audio_arrays = [load_audio(p) for p in batch["audio_path"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=config.SAMPLING_RATE,
            return_tensors=None,
            padding=False,
        )
        batch["input_values"] = inputs["input_values"]
        labels = tokenizer_part(
            batch["transcript"],
            padding=False,
            truncation=False,
        )["input_ids"]
        batch["labels"] = labels
        return batch

    ds_train = _make_hf_dataset(train_df.loc[train_mask])
    ds_val = _make_hf_dataset(train_df.loc[val_mask])
    ds_train = ds_train.map(
        _preprocess, batched=True, remove_columns=["audio_path", "transcript"]
    )
    ds_val = ds_val.map(
        _preprocess, batched=True, remove_columns=["audio_path", "transcript"]
    )

    return (
        DatasetDict({"train": ds_train, "validation": ds_val}),
        feature_extractor,
        tokenizer_part,
    )


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def apply_lora(model, cfg: config.W2VBertConfig):
    """Wrap the model with LoRA adapters. For w2v-BERT we target attention projections."""
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
# Compute Metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(tokenizer):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        pred_strs = tokenizer.batch_decode(pred_ids)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_strs = tokenizer.batch_decode(labels)
        pred_strs = [s.replace(tokenizer.pad_token, "").strip() for s in pred_strs]
        label_strs = [s.replace(tokenizer.pad_token, "").strip() for s in label_strs]
        return compute_metrics(pred_strs, label_strs)

    return _compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_single_fold(fold: int, train_df: pd.DataFrame, cfg: config.W2VBertConfig):
    print(f"\n{'=' * 60}")
    print(f"TRAINING w2v-BERT 2.0 — FOLD {fold + 1}/{config.N_FOLDS}")
    print(f"{'=' * 60}")

    # Load
    print("Loading w2v-BERT 2.0 model...")
    processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCTC.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
    )
    tokenizer = processor.tokenizer

    # LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model, cfg)

    # Data
    print(f"Preparing fold {fold} data...")
    datasets, feature_extractor, tokenizer_part = prepare_dataset_for_fold(
        train_df, fold, processor, tokenizer
    )
    print(f"  Train size: {len(datasets['train'])}")
    print(f"  Val size:   {len(datasets['validation'])}")

    data_collator = DataCollatorCTCWithPadding(feature_extractor, tokenizer_part)

    # Training args
    output_dir = os.path.join(config.CHECKPOINT_DIR, f"w2vbert_fold_{fold}")
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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer_part,
        compute_metrics=make_compute_metrics(tokenizer_part),
    )

    # Enable SpecAugment
    if hasattr(model.config, "apply_spec_augment"):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = cfg.specaug_time_mask
        model.config.mask_feature_prob = 0.0
        model.config.mask_feature_length = cfg.specaug_freq_mask

    print("Starting training...")
    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # Push final adapter to HF Hub (safety — session may time out)
    if cfg.push_to_hub:
        try:
            hub_repo = f"{config.HF_USERNAME}/{cfg.hub_model_id}-fold{fold}"
            print(f"Pushing adapter to HF Hub: {hub_repo} ...")
            model.push_to_hub(hub_repo, token=config.HF_TOKEN)
            print(f"✅ Pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"⚠️  HF Hub push failed: {e}")
            print(f"   Adapter saved locally at {adapter_path} — push manually later.")

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
    cfg = config.W2VBertConfig()

    print(f"Model:      {cfg.model_name}")
    print(f"LoRA r:     {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"Batch size: {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps} accum")

    folds = [args.fold] if args.fold is not None else range(config.N_FOLDS)
    for fold in folds:
        train_single_fold(fold, train_df, cfg)

    print("\nAll folds trained! Checkpoints in:", config.CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
