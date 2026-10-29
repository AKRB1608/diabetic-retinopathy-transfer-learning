"""
Generate corrected report figures for the Diabetic Retinopathy project.

Creates:
  1. three_model_comparison.png
  2. finetuned_mobilenetv2_confusion_matrix.png
  3. project_workflow.png

Requirements:
  pip install matplotlib numpy

Important:
- Metrics are copied from the user's recorded three-model comparison.
- Confusion matrix is the verified fine-tuned MobileNetV2 matrix:
  TN=295, FP=32, FN=11, TP=212.
- Do not use generated illustrative figures as Grad-CAM results. Grad-CAM
  screenshots must come from the actual running application.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("report_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Recorded test results, represented as percentages.
models = [
    "Frozen MobileNetV2",
    "Fine-tuned MobileNetV2",
    "EfficientNetB0 (frozen)",
]
metrics = {
    "Accuracy": [91.09, 92.18, 91.64],
    "Balanced accuracy": [91.37, 92.64, 91.75],
    "Precision": [86.25, 86.89, 87.66],
    "Sensitivity": [92.83, 95.07, 92.38],
    "Specificity": [89.91, 90.21, 91.13],
    "Referable F1": [89.42, 90.79, 89.96],
    "Macro-F1": [90.86, 92.00, 91.40],
    "ROC-AUC": [96.51, 97.29, 97.21],
}

# Figure 1: three-model comparison graph
metric_names = list(metrics.keys())
x = np.arange(len(metric_names))
width = 0.24

fig, ax = plt.subplots(figsize=(14, 7.5))
for i, model in enumerate(models):
    values = metrics[metric_names[i]] if False else [
        metrics[name][i] for name in metric_names
    ]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=model)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, rotation=90)

ax.set_title("Three-Model Comparison on the Same 550-Image Test Set", pad=16)
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=22, ha="right")
ax.set_ylim(75, 101)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right")
fig.text(
    0.01, 0.01,
    "Threshold-dependent metrics use each model's validation-selected threshold. "
    "Results are exploratory, not independent clinical validation.",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "three_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: verified confusion matrix for fine-tuned MobileNetV2.
# Rows = true labels, columns = predicted labels.
cm = np.array([[295, 32], [11, 212]])
fig, ax = plt.subplots(figsize=(7, 5.8))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("Fine-Tuned MobileNetV2 — Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks([0, 1], ["Non-referable (0)", "Referable (1)"])
ax.set_yticks([0, 1], ["Non-referable (0)", "Referable (1)"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=17)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Image count")
fig.text(
    0.01, 0.01,
    "TN=295, FP=32, FN=11, TP=212. Threshold approximately 0.51, selected on validation data.",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(
    OUT / "finetuned_mobilenetv2_confusion_matrix.png",
    dpi=300, bbox_inches="tight",
)
plt.close(fig)

# Figure 3: methodology/workflow diagram (not a Grad-CAM output).
fig, ax = plt.subplots(figsize=(15, 4.6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 4.5)
ax.axis("off")

steps = [
    ("APTOS 2019\nfundus images", 0.3, 2.6, 2.0, 1.1),
    ("Preprocessing\n224 × 224 RGB", 2.8, 2.6, 2.0, 1.1),
    ("Train / validation /\ntest split", 5.3, 2.6, 2.0, 1.1),
    ("Transfer learning\n3 model experiments", 7.8, 2.6, 2.0, 1.1),
    ("Validation threshold\nselection", 10.3, 2.6, 1.8, 1.1),
    ("Test evaluation +\nGrad-CAM display", 12.5, 2.6, 2.1, 1.1),
]
for label, x0, y0, w, h in steps:
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.4, fill=False,
    )
    ax.add_patch(patch)
    ax.text(x0 + w/2, y0 + h/2, label, ha="center", va="center", fontsize=9)
for i in range(len(steps) - 1):
    _, x0, y0, w, h = steps[i]
    _, nx, ny, nw, nh = steps[i + 1]
    ax.add_patch(FancyArrowPatch(
        (x0 + w + 0.05, y0 + h/2),
        (nx - 0.05, ny + nh/2),
        arrowstyle="->", mutation_scale=12, linewidth=1.2,
    ))
ax.text(
    7.5, 1.4,
    "Binary classes: grades 0–1 = non-referable DR  |  grades 2–4 = referable DR",
    ha="center", va="center", fontsize=10,
)
ax.text(
    7.5, 0.75,
    "Exploratory research prototype; not a clinical diagnosis system",
    ha="center", va="center", fontsize=9,
)
ax.set_title("Project Workflow", fontsize=14, pad=10)
fig.tight_layout()
fig.savefig(OUT / "project_workflow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Created corrected report figures in:", OUT.resolve())
for p in sorted(OUT.glob("*.png")):
    print(f" - {p.name} ({p.stat().st_size:,} bytes)")
