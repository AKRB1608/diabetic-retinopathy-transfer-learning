"""
Train a binary diabetic retinopathy classifier using APTOS 2019 labels.
Expected files:
  data/train.csv              columns: id_code, diagnosis
  data/train_images/<id_code>.png (or .jpg/.jpeg)

Binary mapping:
  diagnosis == 0 -> No DR
  diagnosis >= 1 -> DR
"""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
AUTOTUNE = tf.data.AUTOTUNE

ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(r"D:\Projects\diabetic-retinopathy\data\raw\train.csv")
IMAGE_DIR = Path(r"D:\Projects\diabetic-retinopathy\data\raw\train_images")
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def locate_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = IMAGE_DIR / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None

def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}. Place APTOS train.csv there."
        )
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {IMAGE_DIR}. Place the extracted training images there."
        )

    df = pd.read_csv(CSV_PATH)
    required = {"id_code", "diagnosis"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}; found {list(df.columns)}")

    df["path"] = df["id_code"].astype(str).apply(locate_image)
    missing = int(df["path"].isna().sum())
    if missing:
        print(f"Warning: skipping {missing} rows whose images were not found.")
        df = df.dropna(subset=["path"]).copy()

    df["label"] = (df["diagnosis"].astype(int) >= 2).astype(int)
    if df["label"].nunique() != 2:
        raise ValueError("Both classes must be present: diagnosis 0 and diagnosis 1-4.")

    print("\nBinary class counts (0=No DR, 1=DR):")
    print(df["label"].value_counts().sort_index())

    # Stratified splits keep class proportions similar across partitions.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )
    print(f"\nSplit sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    train_ds = make_dataset(train_df["path"].tolist(), train_df["label"].to_numpy(), True)
    val_ds = make_dataset(val_df["path"].tolist(), val_df["label"].to_numpy())
    test_ds = make_dataset(test_df["path"].tolist(), test_df["label"].to_numpy())

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="retinal_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dr_probability")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    train_labels = train_df["label"].to_numpy()
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weight)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_referable_model.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]

    print("\nStage 1: training the classification head...")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
        class_weight=class_weight, callbacks=callbacks
    )

    print("\nStage 2: fine-tuning the last 30 MobileNetV2 layers...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    # Keep BatchNorm layers frozen for stable small-dataset fine-tuning.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight, callbacks=callbacks
    )

    # Load the best validation-AUC checkpoint before final test evaluation.
    best_path = MODEL_DIR / "best_referable_model.keras"
    best_model = tf.keras.models.load_model(best_path)
    probabilities = best_model.predict(test_ds).ravel()
    y_true = test_df["label"].to_numpy()
    y_pred = (probabilities >= 0.5).astype(int)

    metrics = {
        "test_samples": int(len(y_true)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_dr": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_dr": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_dr": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "class_mapping": {"0": "Non-referable DR (grades 0-1)", "1": "Referable DR (grades 2-4)"},
        "note": "Research prototype; not for clinical diagnosis."
    }
    (OUTPUT_DIR / "referable_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "referable_classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=["Non-referable DR (grades 0-1)", "Referable DR (grades 2-4)"], zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Non-referable (grades 0-1)", "Referable (grades 2-4)"])
    ax.set_yticks([0, 1], labels=["Non-referable (grades 0-1)", "Referable (grades 2-4)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Referable DR Confusion Matrix (Test Set)")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "referable_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save split IDs for reproducibility and leakage checks.
    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split[["id_code", "diagnosis", "label"]].to_csv(
            OUTPUT_DIR / f"{name}_split.csv", index=False
        )

    print("\nFINAL TEST METRICS")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print((OUTPUT_DIR / "referable_classification_report.txt").read_text())
    print(f"\nSaved model: {best_path}")
    print(f"Saved metrics and confusion matrix in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()






