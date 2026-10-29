from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

project = Path(r"D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning")
data_root = Path(r"D:\Projects\diabetic-retinopathy\data\raw")
model_path = project / "models" / "best_referable_model.keras"
split_path = project / "outputs" / "test_split.csv"
output_dir = project / "outputs"
threshold = 0.5

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")
if not split_path.exists():
    raise FileNotFoundError(f"Test split not found: {split_path}")

df = pd.read_csv(split_path)
required = {"id_code", "diagnosis"}
if not required.issubset(df.columns):
    raise ValueError(f"Test CSV must contain {required}")

df["diagnosis"] = df["diagnosis"].astype(int)
df["label"] = (df["diagnosis"] >= 2).astype(int)

def find_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = data_root / "train_images" / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image missing for ID: {image_id}")

paths = [find_image(x) for x in df["id_code"].astype(str)]

model = tf.keras.models.load_model(model_path)

from PIL import Image

def load_batch_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((224, 224))
        return np.asarray(img, dtype=np.float32)

images = np.stack([load_batch_image(str(p)) for p in paths])
ds = tf.data.Dataset.from_tensor_slices(images).batch(16)

probabilities = model.predict(ds, verbose=1).ravel()
y_true = df["label"].to_numpy()
y_pred = (probabilities >= threshold).astype(int)

names = ["Non-referable DR (grades 0-1)", "Referable DR (grades 2-4)"]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics = {
    "task": "referable_vs_non_referable_dr",
    "test_samples": int(len(y_true)),
    "threshold": threshold,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision_referable": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall_referable_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
    "specificity_non_referable": float(recall_score(1-y_true, 1-y_pred, zero_division=0)),
    "f1_referable": float(f1_score(y_true, y_pred, zero_division=0)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
    "class_mapping": {
        "0": "Non-referable DR (grades 0-1)",
        "1": "Referable DR (grades 2-4)"
    },
    "note": "Research prototype; not for clinical diagnosis."
}

(output_dir / "referable_metrics.json").write_text(json.dumps(metrics, indent=2))
report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
(output_dir / "referable_classification_report.txt").write_text(report)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="viridis")
ax.set_xticks([0, 1], labels=["Non-referable", "Referable"])
ax.set_yticks([0, 1], labels=["Non-referable", "Referable"])
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Referable DR Confusion Matrix — Test Set")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center", color="white" if cm[i,j] < cm.max()/2 else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(output_dir / "referable_confusion_matrix.png", dpi=180)
plt.close(fig)

np.savez(
    output_dir / "referable_test_predictions.npz",
    y_true=y_true, y_pred=y_pred, probabilities=probabilities
)

print("\nUPDATED TEST METRICS")
print(json.dumps(metrics, indent=2))
print("\nUPDATED CLASSIFICATION REPORT")
print(report)
print("\nUpdated confusion matrix:", output_dir / "referable_confusion_matrix.png")
print("Saved predictions:", output_dir / "referable_test_predictions.npz")

