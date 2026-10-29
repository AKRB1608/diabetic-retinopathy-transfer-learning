
"""Train and evaluate a frozen MobileNetV2 baseline.

Uses the project's existing train/validation/test split CSV files.
Does not overwrite the existing fine-tuned model or its evaluation files.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 12
AUTOTUNE = tf.data.AUTOTUNE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = ROOT / "outputs" / "frozen_baseline"
MODEL_DIR = ROOT / "models"

TRAIN_CSV = ROOT / "outputs" / "train_split.csv"
VAL_CSV = ROOT / "outputs" / "validation_split.csv"
TEST_CSV = ROOT / "outputs" / "test_split.csv"

MODEL_PATH = MODEL_DIR / "best_frozen_baseline.keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_image(image_id):
    """Find an APTOS image using its ID."""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = DATA_DIR / f"{image_id}{ext}"
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        f"Image {image_id!r} not found in {DATA_DIR}"
    )


def read_split(csv_path):
    """Read a saved split and verify its binary labels."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing split file: {csv_path}\n"
            "Do not regenerate splits automatically; check the existing files."
        )

    df = pd.read_csv(csv_path)

    if "id_code" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain id_code")

    if "diagnosis" in df.columns:
        # Referable DR = grades 2, 3, and 4.
        df["label"] = (
            pd.to_numeric(df["diagnosis"], errors="raise").astype(int) >= 2
        ).astype(int)
    elif "label" in df.columns:
        df["label"] = pd.to_numeric(
            df["label"], errors="raise"
        ).astype(int)
    else:
        raise ValueError(
            f"{csv_path.name} must contain diagnosis or label"
        )

    if df["id_code"].duplicated().any():
        raise ValueError(f"Duplicate image IDs found in {csv_path.name}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid binary labels in {csv_path.name}")

    return df.reset_index(drop=True)


def check_splits(train_df, val_df, test_df):
    """Prevent overlap between saved image partitions."""
    train_ids = set(train_df["id_code"].astype(str))
    val_ids = set(val_df["id_code"].astype(str))
    test_ids = set(test_df["id_code"].astype(str))

    if train_ids & val_ids:
        raise ValueError("Train/validation image overlap detected")
    if train_ids & test_ids:
        raise ValueError("Train/test image overlap detected")
    if val_ids & test_ids:
        raise ValueError("Validation/test image overlap detected")

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if df["label"].nunique() != 2:
            raise ValueError(f"{name} split must contain both classes")

        print(
            f"{name.capitalize()}: {len(df)} images; "
            f"class counts={df['label'].value_counts().sort_index().to_dict()}"
        )


def load_image(path, label):
    """Decode and resize RGB images; model performs internal rescaling."""
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(
        raw, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def make_dataset(df, training=False):
    paths = [find_image(str(image_id)) for image_id in df["id_code"]]
    labels = df["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision_referable": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "sensitivity_recall_referable": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "specificity_non_referable": (
            float(tn / (tn + fp)) if tn + fp else None
        ),
        "f1_referable": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "confusion_matrix_labels_0_1": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def choose_validation_threshold(y_true, probabilities):
    """Choose threshold on validation data only."""
    candidates = np.unique(
        np.concatenate([
            np.arange(0.05, 0.951, 0.01),
            np.array([0.5]),
        ])
    )

    scored = [
        (float(t), calculate_metrics(y_true, probabilities, float(t)))
        for t in candidates
    ]

    threshold, best_metrics = max(
        scored,
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["sensitivity_recall_referable"],
            -abs(item[0] - 0.5),
        ),
    )

    return threshold, best_metrics


def save_confusion_matrix(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    ax.set_xticks(
        [0, 1],
        labels=["Non-referable", "Referable"],
    )
    ax.set_yticks(
        [0, 1],
        labels=["Non-referable", "Referable"],
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Frozen MobileNetV2 Baseline — Test Set")

    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    print("=== FROZEN MOBILENETV2 BASELINE ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Dataset directory: {DATA_DIR}")

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {DATA_DIR}"
        )

    # Use the existing partitions; never create new splits here.
    train_df = read_split(TRAIN_CSV)
    val_df = read_split(VAL_CSV)
    test_df = read_split(TEST_CSV)

    check_splits(train_df, val_df, test_df)

    # Record the exact IDs used in this experiment.
    split_ids = {
        name: set(df["id_code"].astype(str))
        for name, df in [
            ("train", train_df),
            ("validation", val_df),
            ("test", test_df),
        ]
    }

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    # Data augmentation runs during training only.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )

    # Pretrained feature extractor; keep every backbone layer frozen.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3), name="retinal_image"
    )
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(
        1.0 / 127.5, offset=-1, name="mobilenet_rescaling"
    )(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="dr_probability"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="frozen_mobilenetv2_baseline")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    train_labels = train_df["label"].to_numpy(dtype=int)
    class_values = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_values,
        y=train_labels,
    )
    class_weight = {
        int(c): float(w) for c, w in zip(class_values, weights)
    }

    print(f"Class weights: {class_weight}")
    print(f"Trainable parameters: {model.count_params()}")
    print("Backbone frozen: ", not base.trainable)
    print("Training classification head...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if not MODEL_PATH.is_file():
        raise RuntimeError("Training ended without saving a baseline model")

    # Load the best checkpoint selected by validation AUC.
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Predicting validation images...")
    y_val = val_df["label"].to_numpy(dtype=int)
    p_val = model.predict(val_ds, verbose=0).reshape(-1)

    print("Selecting threshold using validation data only...")
    threshold, val_metrics = choose_validation_threshold(y_val, p_val)

    print("Predicting test images...")
    y_test = test_df["label"].to_numpy(dtype=int)
    p_test = model.predict(test_ds, verbose=0).reshape(-1)

    # Verify that prediction counts match the saved partitions.
    assert len(p_val) == len(y_val)
    assert len(p_test) == len(y_test)
    assert np.isfinite(p_val).all() and np.isfinite(p_test).all()

    # Save predictions with IDs so later comparisons can verify alignment.
    np.savez_compressed(
        OUTPUT_DIR / "validation_predictions.npz",
        y_true=y_val,
        probabilities=p_val,
        id_code=val_df["id_code"].astype(str).to_numpy(),
    )
    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        y_true=y_test,
        probabilities=p_test,
        id_code=test_df["id_code"].astype(str).to_numpy(),
    )

    test_metrics = calculate_metrics(y_test, p_test, threshold)
    default_test_metrics = calculate_metrics(y_test, p_test, 0.5)

    test_predictions = (p_test >= threshold).astype(int)
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "Non-referable DR (grades 0-1)",
            "Referable DR (grades 2-4)",
        ],
        zero_division=0,
        digits=4,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    save_confusion_matrix(y_test, p_test, threshold)

    # Save the actual experiment configuration and measurements.
    result = {
        "experiment": "frozen_mobilenetv2_baseline",
        "task": "grades_0_1_vs_grades_2_4",
        "model_path": str(MODEL_PATH),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs_requested": HEAD_EPOCHS,
        "epochs_completed": len(history.history.get("loss", [])),
        "backbone": "MobileNetV2 pretrained on ImageNet; all backbone layers frozen",
        "classification_head": "GlobalAveragePooling2D, Dropout(0.30), Dense(1, sigmoid)",
        "preprocessing": {
            "decoder": "TensorFlow decode_image, 3 channels",
            "resize": "tf.image.resize",
            "input_dtype": "float32",
            "pixel_range_before_model_rescaling": "0-255",
            "internal_rescaling": "x / 127.5 - 1",
        },
        "class_weight": class_weight,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "split_overlap_check": "passed",
        "threshold_selection": (
            "Validation macro-F1; sensitivity tie-break; "
            "closest to 0.5 as final tie-break"
        ),
        "selected_threshold": float(threshold),
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_validation_selected_threshold": test_metrics,
        "test_metrics_at_default_threshold_0_5": default_test_metrics,
        "note": (
            "Exploratory research comparison, not clinical validation. "
            "The project's test set was previously examined during development."
        ),
    }

    with open(
        OUTPUT_DIR / "metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)

    print("\n========== FROZEN BASELINE RESULTS ==========")
    print(json.dumps(result, indent=2))
    print("\n========== TEST CLASSIFICATION REPORT ==========")
    print(report)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Results saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
