from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Load the metrics recomputed from your saved model.
val_path = OUT / "validation_metrics_recomputed.json"
test_path = OUT / "test_metrics_recomputed.json"

for path in (val_path, test_path):
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")

with open(val_path, encoding="utf-8") as f:
    val = json.load(f)

with open(test_path, encoding="utf-8") as f:
    test = json.load(f)


def metric(data, *keys):
    """Read a metric using the first available key."""
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    raise KeyError(
        f"Could not find any of {keys} in metrics JSON. "
        f"Available keys: {list(data.keys())}"
    )


# --------------------------------------------------
# FIGURE 3: Stage 1 vs Stage 2 validation ROC-AUC
# Values recorded in the existing report.
# --------------------------------------------------
stages = ["Stage 1", "Stage 2 (selection)", "Final Stage 2"]
auc_values = [0.9636, 0.9736, 0.9731]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(stages, auc_values)

ax.set_title("Validation ROC-AUC Across Training Stages")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.90, 1.00)
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, auc_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.001,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(OUT / "Figure_3_Validation_ROC_AUC.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# FIGURE 4: Validation vs held-out test metrics
# --------------------------------------------------
labels = [
    "Accuracy",
    "Balanced accuracy",
    "Precision",
    "Recall / sensitivity",
    "Specificity",
    "Referable F1",
    "Macro-F1",
    "ROC-AUC",
]

val_values = [
    metric(val, "accuracy"),
    metric(val, "balanced_accuracy"),
    metric(val, "referable_precision"),
    metric(val, "referable_recall"),
    metric(val, "non_referable_specificity"),
    metric(val, "referable_f1"),
    metric(val, "macro_f1"),
    metric(val, "roc_auc"),
]

test_values = [
    metric(test, "accuracy"),
    metric(test, "balanced_accuracy"),
    metric(test, "referable_precision"),
    metric(test, "referable_recall"),
    metric(test, "non_referable_specificity"),
    metric(test, "referable_f1"),
    metric(test, "macro_f1"),
    metric(test, "roc_auc"),
]

x = np.arange(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, val_values, width, label="Validation")
ax.bar(x + width / 2, test_values, width, label="Held-out test")

ax.set_title("Validation vs Held-Out Test Performance")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars in ax.containers:
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "Figure_4_Validation_vs_Test.png", dpi=300)
plt.close(fig)

print("Both figures generated successfully:")
print(OUT / "Figure_3_Validation_ROC_AUC.png")
print(OUT / "Figure_4_Validation_vs_Test.png")
