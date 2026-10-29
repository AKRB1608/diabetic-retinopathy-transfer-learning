
"""Create a reproducible frozen-vs-fine-tuned model comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

FINE_DIR = ROOT / "outputs" / "final_consistent_evaluation"
BASE_DIR = ROOT / "outputs" / "frozen_baseline"

FINE_METRICS = FINE_DIR / "final_consistent_metrics.json"
BASE_METRICS = BASE_DIR / "metrics.json"
FINE_PRED = FINE_DIR / "test_predictions.npz"
BASE_PRED = BASE_DIR / "test_predictions.npz"


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    fine = load_json(FINE_METRICS)
    base = load_json(BASE_METRICS)

    for path in (FINE_PRED, BASE_PRED):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # These are locally generated project prediction files.
    with np.load(FINE_PRED, allow_pickle=True) as f:
        fine_ids = f["id_code"].astype(str)
        fine_y = f["y_true"].astype(int)
        fine_p = f["probabilities"].astype(float)

    with np.load(BASE_PRED, allow_pickle=True) as b:
        base_ids = b["id_code"].astype(str)
        base_y = b["y_true"].astype(int)
        base_p = b["probabilities"].astype(float)

    # Require exact test-set alignment before comparing.
    if not np.array_equal(fine_ids, base_ids):
        raise ValueError("Test image IDs differ or are in a different order.")
    if not np.array_equal(fine_y, base_y):
        raise ValueError("Test labels differ between the two prediction files.")
    if not np.isfinite(fine_p).all() or not np.isfinite(base_p).all():
        raise ValueError("Non-finite probabilities found.")
    if len(fine_y) != 550:
        print(f"Warning: expected 550 test images; found {len(fine_y)}.")

    fine_test = fine["test_metrics_at_validation_selected_threshold"]
    base_test = base["test_metrics_at_validation_selected_threshold"]

    metric_specs = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Referable precision", "precision_referable"),
        ("Referable sensitivity", "sensitivity_recall_referable"),
        ("Non-referable specificity", "specificity_non_referable"),
        ("Referable F1", "f1_referable"),
        ("Macro-F1", "macro_f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    rows = []
    for label, key in metric_specs:
        baseline_value = float(base_test[key])
        fine_value = float(fine_test[key])
        rows.append({
            "Metric": label,
            "Frozen baseline (%)": baseline_value * 100,
            "Fine-tuned model (%)": fine_value * 100,
            "Change (percentage points)": (fine_value - baseline_value) * 100,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison_table.csv", index=False, float_format="%.4f")

    # Save a Markdown table for easy insertion into the report.
    markdown = [
        "# Frozen vs Fine-Tuned MobileNetV2",
        "",
        f"Verified test images: {len(fine_y)}",
        "Test IDs and labels match exactly between both prediction files.",
        "",
        "| Metric | Frozen baseline | Fine-tuned model | Change (pp) |",
        "|---|---:|---:|---:|",
    ]

    for _, row in table.iterrows():
        markdown.append(
            f"| {row['Metric']} | {row['Frozen baseline (%)']:.2f}% "
            f"| {row['Fine-tuned model (%)']:.2f}% "
            f"| {row['Change (percentage points)']:+.2f} |"
        )

    markdown.extend([
        "",
        f"Frozen baseline threshold: {base_test['threshold']:.2f}",
        f"Fine-tuned model threshold: {fine_test['threshold']:.2f}",
        "",
        "Thresholds were selected using each model's validation data. "
        "The test set was not used to choose these thresholds.",
        "",
        "Note: the test set was previously examined during project development. "
        "These results are exploratory and do not constitute independent "
        "clinical validation.",
    ])

    (OUT / "comparison_table.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    # Figure: compare all metrics using a common percentage scale.
    x = np.arange(len(table))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars1 = ax.bar(
        x - width / 2,
        table["Frozen baseline (%)"],
        width,
        label="Frozen baseline",
    )
    bars2 = ax.bar(
        x + width / 2,
        table["Fine-tuned model (%)"],
        width,
        label="Fine-tuned model",
    )

    ax.set_ylabel("Score (%)")
    ax.set_title("Frozen vs Fine-Tuned MobileNetV2 on the Same Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Metric"], rotation=25, ha="right")
    ax.set_ylim(75, 101)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUT / "model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Generate a report-ready Results and Discussion section.
    result_paragraph = f"""## Results and Discussion

A frozen MobileNetV2 baseline was compared with a fine-tuned MobileNetV2
classifier for distinguishing non-referable diabetic retinopathy (grades 0–1)
from referable diabetic retinopathy (grades 2–4). Both models were evaluated
on the same {len(fine_y)} test images, and the saved prediction files were
verified to contain identical image IDs in the same order and matching labels.

The frozen baseline achieved an accuracy of
{base_test['accuracy'] * 100:.2f}%, balanced accuracy of
{base_test['balanced_accuracy'] * 100:.2f}%, macro-F1 of
{base_test['macro_f1'] * 100:.2f}%, and ROC-AUC of
{base_test['roc_auc'] * 100:.2f}%. The fine-tuned model achieved an accuracy
of {fine_test['accuracy'] * 100:.2f}%, balanced accuracy of
{fine_test['balanced_accuracy'] * 100:.2f}%, macro-F1 of
{fine_test['macro_f1'] * 100:.2f}%, and ROC-AUC of
{fine_test['roc_auc'] * 100:.2f}%.

Referable sensitivity was {base_test['sensitivity_recall_referable'] * 100:.2f}%
for the frozen baseline and
{fine_test['sensitivity_recall_referable'] * 100:.2f}% for the fine-tuned
model. The baseline confusion matrix contained
{base_test['FN']} false negatives, while the fine-tuned model contained
{fine_test['FN']}. These observed results favor the fine-tuned model on the
reported metrics.

The models used their own validation-selected thresholds
({base_test['threshold']:.2f} for the baseline and
{fine_test['threshold']:.2f} for the fine-tuned model). Therefore, threshold-
dependent metrics reflect both model behavior and the selected operating
points. The test set had previously been examined during project development,
so this comparison is exploratory rather than an independent evaluation.
External validation on an untouched dataset is needed before drawing
conclusions about generalizability or clinical use.
"""

    (OUT / "results_and_discussion.md").write_text(
        result_paragraph, encoding="utf-8"
    )

    print("\n=== COMPARISON COMPLETE ===")
    print(table.round(2).to_string(index=False))
    print("\nVerified identical test image IDs and labels.")
    print(f"Test images: {len(fine_y)}")
    print(f"Output folder: {OUT}")
    print("Files created:")
    print(" - comparison_table.csv")
    print(" - comparison_table.md")
    print(" - model_comparison.png")
    print(" - results_and_discussion.md")


if __name__ == "__main__":
    main()
