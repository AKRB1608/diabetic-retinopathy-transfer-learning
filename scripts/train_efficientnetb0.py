"""Train and evaluate an EfficientNetB0 transfer-learning model.

Uses the existing project train/validation/test split CSVs and does not
overwrite existing MobileNetV2 checkpoints or evaluation artifacts.

Binary task:
  0 = non-referable DR (original grades 0-1)
  1 = referable DR (original grades 2-4)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
)

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 12
AUTOTUNE = tf.data.AUTOTUNE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = ROOT / "outputs" / "efficientnetb0"
MODEL_DIR = ROOT / "models"
TRAIN_CSV = ROOT / "outputs" / "train_split.csv"
VAL_CSV = ROOT / "outputs" / "validation_split.csv"
TEST_CSV = ROOT / "outputs" / "test_split.csv"
MODEL_PATH = MODEL_DIR / "best_efficientnetb0.keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = DATA_DIR / f"{image_id}{ext}"
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Image {image_id!r} not found in {DATA_DIR}")


def read_split(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain id_code")
    if "diagnosis" in df.columns:
        df["label"] = (
            pd.to_numeric(df["diagnosis"], errors="raise").astype(int) >= 2
        ).astype(int)
    elif "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    else:
        raise ValueError(f"{csv_path.name} must contain diagnosis or label")
    if df["id_code"].duplicated().any():
        raise ValueError(f"Duplicate image IDs found in {csv_path.name}")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid binary labels in {csv_path.name}")
    return df.reset_index(drop=True)


def check_splits(train_df, val_df, test_df):
    partitions = {
        "train": set(train_df["id_code"].astype(str)),
        "validation": set(val_df["id_code"].astype(str)),
        "test": set(test_df["id_code"].astype(str)),
    }
    if partitions["train"] & partitions["validation"]:
        raise ValueError("Train/validation image overlap detected")
    if partitions["train"] & partitions["test"]:
        raise ValueError("Train/test image overlap detected")
    if partitions["validation"] & partitions["test"]:
        raise ValueError("Validation/test image overlap detected")
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if df["label"].nunique() != 2:
            raise ValueError(f"{name} split must contain both classes")
        print(
            f"{name.capitalize()}: {len(df)} images; "
            f"class counts={df['label'].value_counts().sort_index().to_dict()}"
        )


def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    # Keras EfficientNetB0 includes its own rescaling/preprocessing layer.
    # Keep pixel values in the [0, 255] range; do not apply preprocess_input
    # or divide by 255 here.
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def make_dataset(df, training=False):
    paths = [find_image(str(image_id)) for image_id in df["id_code"]]
    labels = df["label"].to_numpy(dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048), seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_referable": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity_recall_referable": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity_non_referable": float(tn / (tn + fp)) if tn + fp else None,
        "f1_referable": float(f1_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def choose_validation_threshold(y_true, probabilities):
    candidates = np.unique(np.concatenate([np.arange(0.05, 0.951, 0.01), [0.5]]))
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


def main():
    print("TensorFlow:", tf.__version__)
    train_df = read_split(TRAIN_CSV)
    val_df = read_split(VAL_CSV)
    test_df = read_split(TEST_CSV)
    check_splits(train_df, val_df, test_df)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    print("\nBuilding EfficientNetB0...")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3), pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="fundus_image")
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.25, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="referable_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_referable_dr")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes,
        y=train_df["label"].to_numpy(dtype=int),
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weights)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_history.csv")),
    ]

    print("\nTraining frozen EfficientNetB0 feature extractor and classifier head...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )
    # Load the best validation-AUC checkpoint, not necessarily the final epoch.
    model = tf.keras.models.load_model(MODEL_PATH)

    def probabilities(ds):
        return model.predict(ds, verbose=1).reshape(-1).astype(float)

    val_y = val_df["label"].to_numpy(dtype=int)
    test_y = test_df["label"].to_numpy(dtype=int)
    val_p = probabilities(val_ds)
    test_p = probabilities(test_ds)

    threshold, val_metrics = choose_validation_threshold(val_y, val_p)
    test_metrics = calculate_metrics(test_y, test_p, threshold)

    val_pred = (val_p >= threshold).astype(int)
    test_pred = (test_p >= threshold).astype(int)
    metrics = {
        "task": "non_referable_grades_0_1_vs_referable_grades_2_4",
        "model": "EfficientNetB0 transfer learning (frozen ImageNet backbone)",
        "image_size": list(IMAGE_SIZE),
        "threshold_selection": "validation macro-F1; sensitivity tie-break; closest to 0.5",
        "validation_selected_threshold": float(threshold),
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_validation_selected_threshold": test_metrics,
        "note": (
            "Exploratory project results. The test set has been examined during "
            "development and is not an independent clinical validation set."
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "threshold.json").write_text(
        json.dumps({"threshold": float(threshold), "selection_data": "validation"}, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_DIR / "validation_predictions.npz",
        id_code=val_df["id_code"].astype(str).to_numpy(),
        y_true=val_y, probabilities=val_p, predictions=val_pred,
    )
    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        id_code=test_df["id_code"].astype(str).to_numpy(),
        y_true=test_y, probabilities=test_p, predictions=test_pred,
    )
    report = classification_report(
        test_y, test_pred, labels=[0, 1],
        target_names=["Non-referable DR (grades 0-1)", "Referable DR (grades 2-4)"],
        digits=4, zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = np.array(test_metrics["confusion_matrix_labels_0_1"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)
    ax.set_title(f"EfficientNetB0 Test Confusion Matrix (threshold={threshold:.2f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Non-referable", "Referable"])
    ax.set_yticks([0, 1], ["Non-referable", "Referable"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== EFFICIENTNETB0 COMPLETE ===")
    print("Saved model:", MODEL_PATH)
    print("Output folder:", OUTPUT_DIR)
    print("Validation-selected threshold:", round(threshold, 4))
    print("\nTest metrics (threshold selected on validation only):")
    print(json.dumps(test_metrics, indent=2))
    print("\nClassification report:\n", report)
    print("\nGenerated files: metrics.json, threshold.json, training_history.csv,")
    print("validation_predictions.npz, test_predictions.npz, classification_report.txt,")
    print("confusion_matrix.png")


if __name__ == "__main__":
    main()
