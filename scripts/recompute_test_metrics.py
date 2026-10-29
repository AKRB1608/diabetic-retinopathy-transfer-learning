from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "outputs" / "test_split.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "train_images"
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
THRESHOLD_PATH = ROOT / "outputs" / "referable_threshold.json"
OUT_PATH = ROOT / "outputs" / "test_metrics_recomputed.json"

for path in (CSV_PATH, IMAGE_DIR, MODEL_PATH, THRESHOLD_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

df = pd.read_csv(CSV_PATH)
if not {"id_code", "label"}.issubset(df.columns):
    raise ValueError(f"Unexpected test CSV columns: {list(df.columns)}")

threshold = float(json.loads(
    THRESHOLD_PATH.read_text(encoding="utf-8")
)["threshold"])

def find_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = IMAGE_DIR / f"{image_id}{ext}"
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Image missing for ID: {image_id}")

paths = [find_image(str(x)) for x in df["id_code"]]
y_true = df["label"].astype(int).to_numpy()

def load_image(path):
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (224, 224))
    return tf.cast(image, tf.float32)

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(MODEL_PATH)
prob = model.predict(ds, verbose=1).reshape(-1)

if len(prob) != len(y_true):
    raise RuntimeError("Test prediction count does not match labels.")

pred = (prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_true, pred, labels=[0, 1]
).ravel()

results = {
    "test_samples": int(len(y_true)),
    "threshold": threshold,
    "accuracy": float(accuracy_score(y_true, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    "referable_precision": float(precision_score(
        y_true, pred, zero_division=0
    )),
    "referable_recall": float(recall_score(
        y_true, pred, zero_division=0
    )),
    "non_referable_specificity": float(
        tn / (tn + fp) if tn + fp else 0
    ),
    "referable_f1": float(f1_score(y_true, pred, zero_division=0)),
    "macro_f1": float(f1_score(
        y_true, pred, average="macro", zero_division=0
    )),
    "roc_auc": float(roc_auc_score(y_true, prob)),
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "model": str(MODEL_PATH),
    "note": "Recomputed on the held-out test split; research use only."
}

OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n=== RECOMPUTED HELD-OUT TEST METRICS ===")
print(json.dumps(results, indent=2))
print("\n=== TEST CLASSIFICATION REPORT ===")
print(classification_report(
    y_true, pred, labels=[0, 1],
    target_names=["Non-referable", "Referable"],
    zero_division=0
))
print(f"\nSaved to: {OUT_PATH}")
