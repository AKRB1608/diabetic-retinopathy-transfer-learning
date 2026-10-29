from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score, recall_score, confusion_matrix

ROOT = Path(r"D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning")
DATA = Path(r"D:\Projects\diabetic-retinopathy\data\raw")
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
SPLIT_PATH = ROOT / "outputs" / "validation_split.csv"
OUT_PATH = ROOT / "outputs" / "referable_threshold.json"
SIZE = (224, 224)
BATCH_SIZE = 16

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not SPLIT_PATH.exists():
    raise FileNotFoundError(SPLIT_PATH)

df = pd.read_csv(SPLIT_PATH)
required = {"id_code", "diagnosis", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"Expected {required}; got {list(df.columns)}")

# Confirm the intended mapping: grades 0-1 -> 0; grades 2-4 -> 1.
expected = (df["diagnosis"].astype(int) >= 2).astype(int)
if not np.array_equal(expected.to_numpy(), df["label"].astype(int).to_numpy()):
    raise ValueError("Validation labels do not match grades 0-1 vs grades 2-4.")

def find_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = DATA / "train_images" / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for ID: {image_id}")

paths = [find_image(i) for i in df["id_code"].astype(str)]
images = []
for path in paths:
    with Image.open(path) as im:
        im = im.convert("RGB").resize(SIZE)
        images.append(np.asarray(im, dtype=np.float32))

x = np.stack(images)
y = df["label"].astype(int).to_numpy()

print("Loading existing model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"Predicting {len(y)} validation images...")
prob = model.predict(x, batch_size=BATCH_SIZE, verbose=1).reshape(-1)

if not np.isfinite(prob).all():
    raise ValueError("Model produced non-finite predictions.")

# Choose threshold for best validation Macro-F1, with referable recall
# as a tie-breaker. This is a research operating point, not a clinical cutoff.
candidates = np.arange(0.10, 0.901, 0.01)
rows = []
for t in candidates:
    pred = (prob >= t).astype(int)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    sensitivity = recall_score(y, pred, pos_label=1, zero_division=0)
    rows.append((float(t), float(macro_f1), float(sensitivity)))

best = max(rows, key=lambda r: (r[1], r[2], -abs(r[0] - 0.5)))
threshold, macro_f1, sensitivity = best
pred = (prob >= threshold).astype(int)

result = {
    "threshold": threshold,
    "selection_method": "maximum validation macro-F1; referable recall tie-break",
    "validation_samples": int(len(y)),
    "validation_macro_f1": macro_f1,
    "validation_referable_recall": sensitivity,
    "validation_confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    "class_mapping": {
        "0": "Non-referable DR (grades 0-1)",
        "1": "Referable DR (grades 2-4)"
    },
    "note": "Research operating threshold selected on validation data; not a clinical cutoff."
}
OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("\nVALIDATION THRESHOLD")
print(json.dumps(result, indent=2))
print(f"\nSaved: {OUT_PATH}")
