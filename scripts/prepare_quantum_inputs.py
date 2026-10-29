from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "quantum_experiment" / "features"
OUTPUT_DIR = ROOT / "quantum_experiment" / "prepared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name):
    path = FEATURE_DIR / f"{name}_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    data = np.load(path, allow_pickle=False)
    return data["features"].astype(np.float32), data["labels"].astype(np.int32)

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

print("Raw shapes:", X_train.shape, X_val.shape, X_test.shape)

# Fit transformations on TRAINING DATA ONLY.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=4, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Map PCA coordinates into rotation-angle inputs in [-pi, pi].
# The mapping is calibrated using training-set component ranges only.
train_min = X_train_pca.min(axis=0)
train_max = X_train_pca.max(axis=0)
train_range = np.maximum(train_max - train_min, 1e-8)

def to_angles(X):
    scaled = 2.0 * (X - train_min) / train_range - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    return (scaled * np.pi).astype(np.float32)

splits = {
    "train": (to_angles(X_train_pca), y_train),
    "validation": (to_angles(X_val_pca), y_val),
    "test": (to_angles(X_test_pca), y_test),
}

for name, (features, labels) in splits.items():
    out = OUTPUT_DIR / f"{name}_quantum_inputs.npz"
    np.savez_compressed(out, features=features, labels=labels)
    print(
        f"{name:10s}: features={features.shape}, "
        f"labels={labels.shape}, "
        f"angle_range=({features.min():.3f}, {features.max():.3f})"
    )
    print("Saved:", out)

# Save fitted transforms and training-derived angle bounds for reproducibility.
import joblib
joblib.dump(scaler, OUTPUT_DIR / "standard_scaler.joblib")
joblib.dump(pca, OUTPUT_DIR / "pca_4_components.joblib")
np.savez(
    OUTPUT_DIR / "angle_bounds.npz",
    train_min=train_min,
    train_max=train_max,
)

metadata = {
    "input_features": int(X_train.shape[1]),
    "pca_components": 4,
    "qubits_planned": 4,
    "training_samples": int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "test_samples": int(len(y_test)),
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
    "fit_on_training_split_only": True,
}
(OUTPUT_DIR / "preprocessing_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", round(metadata["explained_variance_sum"], 4))
print("FOUR-INPUT PREPARATION COMPLETED")
