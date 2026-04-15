"""
Train a model on Golang coding tasks using QLoRA + PEFT.

All settings are read from .env via config.py.

Usage:
    python train.py
    python train.py --epochs 3
"""

import argparse

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

import config as cfg
from core.model import load_base_model, load_tokenizer, apply_lora
from core.formatting import format_chatml


def format_for_sft(example: dict) -> dict:
    """Convert messages list to a single text string for SFT."""
    return {"text": format_chatml(example.get("messages", []))}


def main():
    parser = argparse.ArgumentParser(description="Train Go coding model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    epochs = args.epochs or cfg.NUM_TRAIN_EPOCHS
    lr = args.lr or cfg.LEARNING_RATE

    print("=" * 60)
    print(f"Training: {cfg.MODEL_NAME}")
    print(f"LoRA r={cfg.LORA_R}, seq_len={cfg.MAX_SEQ_LENGTH}, epochs={epochs}")
    print("=" * 60)

    # Load model + LoRA
    model = load_base_model()
    tokenizer = load_tokenizer()
    model = apply_lora(model)

    # Load dataset
    dataset = load_dataset("json", data_files=str(cfg.DATASET_PATH), split="train")
    dataset = dataset.map(format_for_sft)

    if len(dataset) >= 10:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = dataset, dataset

    print(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    # Trainer
    tokenizer.model_max_length = cfg.MAX_SEQ_LENGTH

    training_args = SFTConfig(
        output_dir=str(cfg.LORA_DIR),
        per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.WARMUP_STEPS,
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=cfg.SAVE_STEPS,
        eval_strategy="epoch",
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        bf16=True,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    stats = trainer.train(resume_from_checkpoint=args.resume)

    # Save
    model.save_pretrained(str(cfg.LORA_DIR))
    tokenizer.save_pretrained(str(cfg.LORA_DIR))

    print("\n" + "=" * 60)
    print(f"Training complete! {stats.metrics['train_runtime']:.0f}s")
    print(f"Model saved to: {cfg.LORA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
