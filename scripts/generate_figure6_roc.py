
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# Project paths
ROOT = Path(__file__).resolve().parent
PREDICTIONS = (
    ROOT / "outputs" / "final_consistent_evaluation"
    / "test_predictions.npz"
)
OUTPUT_DIR = ROOT / "outputs" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = OUTPUT_DIR / "fig06_test_roc_curve.png"
METRICS_PATH = OUTPUT_DIR / "fig06_test_roc_metrics.json"

# Load saved held-out test predictions
if not PREDICTIONS.exists():
    raise FileNotFoundError(
        f"Predictions file not found:\n{PREDICTIONS}\n"
        "Run the final consistent evaluation script first."
    )

with np.load(PREDICTIONS, allow_pickle=False) as data:
    print("Available arrays:", data.files)

    if "y_true" not in data.files:
        raise KeyError("The NPZ file does not contain 'y_true'.")

    y_true = np.asarray(data["y_true"]).reshape(-1)

    if "probabilities" in data.files:
        scores = np.asarray(data["probabilities"]).reshape(-1)
    elif "y_prob" in data.files:
        scores = np.asarray(data["y_prob"]).reshape(-1)
    else:
        raise KeyError(
            "No probability array found. Expected 'probabilities' "
            "or 'y_prob'. ROC curves require probability scores."
        )

# Validate arrays
if len(y_true) != len(scores):
    raise ValueError(
        f"Label count ({len(y_true)}) does not match "
        f"probability count ({len(scores)})."
    )

if not np.isin(y_true, [0, 1]).all():
    raise ValueError("Expected binary labels: 0 and 1.")

if not np.isfinite(scores).all():
    raise ValueError("Probability scores contain NaN or infinity.")

if len(np.unique(y_true)) != 2:
    raise ValueError("ROC-AUC requires both classes in the test set.")

# Calculate ROC curve and AUC from continuous scores
fpr, tpr, thresholds = roc_curve(y_true, scores)
auc_value = roc_auc_score(y_true, scores)

# Publication-quality plot
plt.figure(figsize=(8, 6), dpi=150)
plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"MobileNetV2 (AUC = {auc_value:.4f})",
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="No-skill baseline (AUC = 0.5000)",
)

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title(
    "ROC Curve — Binary Referable Diabetic Retinopathy",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.25)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()

plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close()

# Save the measured AUC for report consistency
results = {
    "figure": "Figure 6 — Held-out test ROC curve",
    "task": "Non-referable grades 0-1 vs referable grades 2-4",
    "test_samples": int(len(y_true)),
    "roc_auc": float(auc_value),
    "source_predictions": str(PREDICTIONS),
    "curve_uses_continuous_probabilities": True,
}

METRICS_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\n========== FIGURE 6 GENERATED ==========")
print(f"Test samples : {len(y_true)}")
print(f"ROC-AUC      : {auc_value:.6f}")
print(f"Figure       : {FIGURE_PATH}")
print(f"Metrics      : {METRICS_PATH}")
