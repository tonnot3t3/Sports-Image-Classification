"""Fine-tune google/vit-base-patch16-224 on the Kaggle Sports Classification dataset.

The dataset is expected to be unzipped under ``./sports_dataset`` with
the standard Kaggle layout::

    sports_dataset/
        train/<class_name>/*.jpg
        valid/<class_name>/*.jpg
        test/<class_name>/*.jpg
        sports.csv

Usage:
    pip install -r requirements-dev.txt
    python scripts/train.py \\
        --data_dir ./sports_dataset \\
        --output_dir ./vit_sports_finetuned \\
        --epochs 5 --batch_size 32 --lr 5e-5

The script saves a checkpoint that ``scripts/optimize.py`` consumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", default="google/vit-base-patch16-224")
    p.add_argument("--data_dir", default="./sports_dataset")
    p.add_argument("--output_dir", default="./vit_sports_finetuned")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use mixed precision (CUDA only).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

def build_transforms(processor: ViTImageProcessor):
    """Train-time augmentations + ImageNet-style normalization for ViT."""
    size = processor.size["height"]
    mean = processor.image_mean
    std = processor.image_std

    train_tx = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tx = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tx, eval_tx


def collate(batch):
    """Collate a list of per-sample dicts into a batch dict for the Trainer."""
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def make_batch_transform(tx, image_col: str):
    """Wrap a torchvision pipeline so it works with HF datasets' batched transform.

    `with_transform`/`set_transform` in datasets >= 2.x always call the
    transform with a batched dict (each value is a list).
    """
    def _fn(examples):
        examples["pixel_values"] = [
            tx(img.convert("RGB")) for img in examples[image_col]
        ]
        return examples
    return _fn


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(eval_pred):
    """Accuracy + macro-F1 in pure NumPy (no scikit-learn dependency)."""
    import numpy as np

    preds = np.asarray(eval_pred.predictions).argmax(axis=-1)
    labels = np.asarray(eval_pred.label_ids)

    accuracy = float((preds == labels).mean()) if labels.size else 0.0

    classes = np.unique(np.concatenate([labels, preds]))
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom else 0.0)
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    return {"accuracy": accuracy, "f1_macro": f1_macro}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ---- Dataset (Kaggle ImageFolder layout) -------------------------------
    # Load each split via its own folder with the standard `data_dir=` path
    # (the `data_files=` glob form does NOT populate the image column in
    # datasets >= 4.x).
    print(f"Loading dataset from {data_dir} ...")
    split_map = [("train", "train"), ("valid", "validation"), ("test", "test")]
    dset = DatasetDict()
    for kaggle_split, hf_split in split_map:
        split_dir = data_dir / kaggle_split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected Kaggle split folder not found: {split_dir}"
            )
        loaded = load_dataset("imagefolder", data_dir=str(split_dir))
        # `imagefolder` always produces a single 'train' split inside the
        # returned DatasetDict regardless of source.
        dset[hf_split] = loaded["train"]

    cols = dset["train"].column_names
    print(f"Dataset columns: {cols}")
    image_col = "image" if "image" in cols else next(c for c in cols if c != "label")
    print(f"Using image column: {image_col!r}")

    label_names = dset["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}
    print(f"Detected {len(label_names)} classes.")

    # ---- Save labels.json so the API + ONNX export agree on indices --------
    labels_out = Path("app") / "labels.json"
    labels_out.write_text(json.dumps(label_names, indent=2))
    print(f"Wrote {labels_out}")

    # ---- Processor + transforms --------------------------------------------
    processor = ViTImageProcessor.from_pretrained(args.model_id)
    train_tx, eval_tx = build_transforms(processor)

    # ใช้ make_batch_transform ที่เตรียมไว้ด้านบน และส่งตัวแปร image_col เข้าไป
    dset["train"] = dset["train"].with_transform(make_batch_transform(train_tx, image_col))
    dset["validation"] = dset["validation"].with_transform(make_batch_transform(eval_tx, image_col))
    dset["test"] = dset["test"].with_transform(make_batch_transform(eval_tx, image_col))

    # ---- Model --------------------------------------------------------------
    model = ViTForImageClassification.from_pretrained(
        args.model_id,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ---- TrainingArguments --------------------------------------------------
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        # Renamed in transformers 4.46 (was `evaluation_strategy`).
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dset["train"],
        eval_dataset=dset["validation"],
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- Final test-set evaluation -----------------------------------------
    print("Evaluating on held-out test split ...")
    test_metrics = trainer.evaluate(eval_dataset=dset["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # ---- Persist final model + processor for downstream optimization -------
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
