"""
RetinaAI report figure generator
Run from PowerShell:
  python generate_report_figures.py
Optional:
  python generate_report_figures.py --project "D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning"

This script uses saved project outputs. It does NOT invent Stage 1/Stage 2 values.
Figures are saved to <project>\outputs\report_figures.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, roc_curve
)

DEFAULT_PROJECT = r"D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning"

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def save_bar(labels, values, title, ylabel, path, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()
    root = Path(args.project)
    outputs = root / "outputs"
    out = outputs / "report_figures"
    out.mkdir(parents=True, exist_ok=True)

    npz_path = first_existing([
        outputs / "referable_test_predictions.npz",
        outputs / "test_predictions.npz",
    ])
    test_csv = first_existing([
        outputs / "test_split.csv",
        outputs / "test.csv",
    ])
    threshold_path = first_existing([
        outputs / "referable_threshold.json",
        outputs / "threshold.json",
    ])
    metrics_path = first_existing([
        outputs / "referable_metrics.json",
        outputs / "metrics.json",
    ])

    print("Project:", root)
    print("Figures folder:", out)

    # Figure 3: only draw Stage 1 vs Stage 2 when real values are found in saved metrics.
    # This prevents fabricating results if metric keys differ or values are absent.
    if metrics_path:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
        def find_number(obj, candidates):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k).lower().replace("-", "_").replace(" ", "_")
                    if any(c in key for c in candidates) and isinstance(v, (int, float)):
                        return float(v)
                for v in obj.values():
                    found = find_number(v, candidates)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for v in obj:
                    found = find_number(v, candidates)
                    if found is not None:
                        return found
            return None

        stage1 = find_number(metrics, ["stage1_val_auc", "stage_1_val_auc", "stage1_validation_auc", "stage_1_validation_auc"])
        stage2 = find_number(metrics, ["stage2_val_auc", "stage_2_val_auc", "stage2_validation_auc", "stage_2_validation_auc"])
        if stage1 is not None and stage2 is not None:
            save_bar(["Stage 1", "Stage 2"], [stage1, stage2],
                     "Validation ROC-AUC: Stage 1 vs Stage 2", "ROC-AUC",
                     out / "fig03_stage_comparison.png", (0, 1.05))
            print("Created fig03_stage_comparison.png using metric values in", metrics_path.name)
        else:
            print("SKIP Fig. 3: Stage 1/Stage 2 validation AUC keys were not found in metrics JSON.")
            print("Use the actual training log/CSV values; do not type guessed values.")

    if not npz_path:
        print("ERROR: saved predictions NPZ not found. Expected outputs/referable_test_predictions.npz")
        print("No prediction-derived figures were generated.")
        return

    data = np.load(npz_path)
    required = {"y_true", "probabilities"}
    if not required.issubset(set(data.files)):
        print("ERROR: NPZ must contain y_true and probabilities. Found:", list(data.files))
        return
    y_true = np.asarray(data["y_true"]).astype(int).ravel()
    prob = np.asarray(data["probabilities"]).astype(float).ravel()

    if len(y_true) != len(prob):
        print("ERROR: y_true/probabilities length mismatch:", len(y_true), len(prob))
        return
    if test_csv and len(pd.read_csv(test_csv)) != len(y_true):
        print("WARNING: test CSV row count does not match saved predictions. Do not use these figures until row alignment is verified.")
        print("CSV rows:", len(pd.read_csv(test_csv)), "predictions:", len(y_true))
    else:
        print("Saved predictions:", len(y_true))

    threshold = 0.5
    if threshold_path:
        try:
            saved = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(saved.get("threshold", threshold))
        except Exception as e:
            print("Could not read threshold JSON; using 0.50 only as an explicitly labelled comparison:", e)

    # Apply the currently saved research threshold to probabilities, not stale y_pred.
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")

    # Figure 4: measured metric chart for current saved test predictions.
    names = ["Accuracy", "Balanced acc.", "Precision", "Sensitivity", "Specificity", "Referable F1"]
    vals = [accuracy, balanced, precision, sensitivity, specificity, f1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Test-set metrics at saved threshold {threshold:.2f}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig04_test_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array([[tn, fp], [fn, tp]]),
        display_labels=["Non-referable (0)", "Referable (1)"]
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Test Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out / "fig05_test_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: ROC curve from saved probabilities.
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, prob, ax=ax, name=f"Saved test predictions (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC Curve — Binary Referable DR")
    fig.tight_layout()
    fig.savefig(out / "fig06_test_roc_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Additional useful figure: class counts in the actual test split.
    labels, counts = np.unique(y_true, return_counts=True)
    class_names = [{0: "Non-referable (0)", 1: "Referable (1)"}.get(int(k), str(k)) for k in labels]
    save_bar(class_names, counts, "Test Split Class Distribution", "Number of images",
             out / "optional_test_class_distribution.png")

    # Machine-readable summary to copy verified numbers into the report.
    summary = {
        "source_npz": str(npz_path),
        "source_test_csv": str(test_csv) if test_csv else None,
        "threshold_source": str(threshold_path) if threshold_path else "default 0.50 (not saved)",
        "threshold_used": threshold,
        "n_test": int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "referable_precision": float(precision),
        "referable_sensitivity": float(sensitivity),
        "non_referable_specificity": float(specificity),
        "referable_f1": float(f1),
        "roc_auc": float(auc),
        "warning": "Verify prediction row alignment with test_split.csv before using results in a report."
    }
    (out / "verified_test_metrics_from_saved_predictions.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nSaved figures:")
    for p in sorted(out.glob("*.png")):
        print(" -", p.name)
    print("\nRecomputed values (audit alignment before reporting):")
    print(json.dumps(summary, indent=2))
    print("\nFigures 7–10 are actual application screenshots/Grad-CAM outputs and must be captured from the running app.")

if __name__ == "__main__":
    main()
