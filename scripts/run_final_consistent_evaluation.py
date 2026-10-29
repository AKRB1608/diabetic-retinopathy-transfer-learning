import os, json, hashlib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)

ROOT = Path(r"D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning")
OUT = ROOT / "outputs" / "final_consistent_evaluation"
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
DATA_DIR = ROOT / "data" / "raw" / "train_images"
SIZE = (224, 224)
BATCH = 16

def find_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = DATA_DIR / f"{image_id}{ext}"
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Image not found: {image_id}")

def read_split(filename):
    path = ROOT / "outputs" / filename
    df = pd.read_csv(path)
    if "diagnosis" in df.columns:
        df["label"] = (df["diagnosis"].astype(int) >= 2).astype(int)
    elif "label" not in df.columns:
        raise ValueError(f"{filename} needs diagnosis or label column")
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {filename}")
    return df

def predict_split(model, df, name):
    paths = [find_image(str(x)) for x in df["id_code"]]
    y = df["label"].to_numpy(dtype=int)

    def load_image(path):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, SIZE)
        return tf.cast(image, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    prob = model.predict(ds, verbose=0).reshape(-1)

    if len(prob) != len(y):
        raise ValueError(f"{name}: prediction/label count mismatch")
    if not np.isfinite(prob).all():
        raise ValueError(f"{name}: non-finite probabilities")
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError(
            f"{name}: model output isn't a probability in [0,1]; "
            "inspect the final model layer before proceeding"
        )
    np.savez_compressed(
        OUT / f"{name.lower()}_predictions.npz",
        y_true=y, probabilities=prob,
        id_code=df["id_code"].astype(str).to_numpy()
    )
    return y, prob

def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_referable": float(precision_score(y, pred, zero_division=0)),
        "sensitivity_recall_referable": float(recall_score(y, pred, zero_division=0)),
        "specificity_non_referable": float(tn / (tn + fp)) if tn + fp else None,
        "f1_referable": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    model = tf.keras.models.load_model(MODEL_PATH)

    val_df = read_split("validation_split.csv")
    test_df = read_split("test_split.csv")

    print("Generating validation predictions with training-compatible preprocessing...")
    yv, pv = predict_split(model, val_df, "validation")
    print("Generating held-out test predictions...")
    yt, pt = predict_split(model, test_df, "test")

    # Choose threshold ONLY on validation data; never optimize on test labels.
    candidates = np.unique(np.concatenate([
        np.arange(0.05, 0.951, 0.01), np.array([0.5])
    ]))
    rows = [(float(t), metrics(yv, pv, t)) for t in candidates]
    best_threshold, best_val = max(
        rows, key=lambda x: (
            x[1]["macro_f1"],
            x[1]["sensitivity_recall_referable"],
            -abs(x[0] - 0.5)
        )
    )

    result = {
        "task": "non_referable_grades_0_1_vs_referable_grades_2_4",
        "model": str(MODEL_PATH),
        "model_sha256": model_hash,
        "preprocessing": {
            "decoder": "TensorFlow decode_image, 3 channels",
            "resize": "tf.image.resize",
            "size": [224, 224],
            "input_dtype": "float32",
            "input_pixel_range_before_model_rescaling": "0-255",
            "model_internal_rescaling": "verify saved architecture; expected 1/127.5 offset -1"
        },
        "threshold_selection": "validation macro-F1; sensitivity tie-break; no test-set threshold tuning",
        "selected_threshold": best_threshold,
        "validation_metrics_at_selected_threshold": best_val,
        "test_metrics_at_validation_selected_threshold": metrics(yt, pt, best_threshold),
        "test_metrics_at_default_threshold_0_5": metrics(yt, pt, 0.5),
        "note": "Held-out test results are reported, not used to select the threshold."
    }

    with open(OUT / "final_consistent_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n========== FINAL CONSISTENT EVALUATION ==========")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {OUT}")
    print("Original metric files were not overwritten.")

if __name__ == "__main__":
    main()
