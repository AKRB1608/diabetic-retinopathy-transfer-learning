
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
TEST_CSV = OUTPUT_DIR / "test_split.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "train_images"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Check required files
for path in (MODEL_PATH, TEST_CSV, IMAGE_DIR):
    if not path.exists():
        raise FileNotFoundError(f"Required file/folder not found: {path}")

# Load held-out test split and saved model
test_df = pd.read_csv(TEST_CSV)
model = tf.keras.models.load_model(MODEL_PATH)

def locate_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Image not found for ID: {image_id}")

test_df["path"] = test_df["id_code"].astype(str).apply(locate_image)

# Use the saved binary labels: 0 = non-referable, 1 = referable
if "label" not in test_df.columns:
    test_df["label"] = (test_df["diagnosis"].astype(int) >= 2).astype(int)

def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(
        raw, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label

ds = tf.data.Dataset.from_tensor_slices(
    (test_df["path"].to_numpy(), test_df["label"].to_numpy())
)
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predict probabilities and preserve dataset order
y_true = test_df["label"].to_numpy()
y_score = model.predict(ds, verbose=1).ravel()

if len(y_true) != len(y_score):
    raise ValueError("Test labels and predictions have different lengths.")

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

# Save ROC curve
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(fpr, tpr, linewidth=2,
        label=f"Saved test predictions (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--",
        linewidth=1.5, label="No-skill baseline")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve — Binary Referable Diabetic Retinopathy")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()

save_path = OUTPUT_DIR / "Figure_ROC_Curve_Test.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Test ROC-AUC: {auc:.4f}")
print(f"ROC curve saved to: {save_path}")
