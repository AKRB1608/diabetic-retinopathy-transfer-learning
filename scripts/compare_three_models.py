"""Compare frozen MobileNetV2, fine-tuned MobileNetV2, and EfficientNetB0.

Requires prediction and metrics artifacts produced by the existing project scripts.
All three prediction files must have exactly aligned test IDs and labels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "label": "Frozen MobileNetV2",
        "metrics": ROOT / "outputs" / "frozen_baseline" / "metrics.json",
        "predictions": ROOT / "outputs" / "frozen_baseline" / "test_predictions.npz",
        "metric_key": "test_metrics_at_validation_selected_threshold",
    },
    {
        "label": "Fine-tuned MobileNetV2",
        "metrics": ROOT / "outputs" / "final_consistent_evaluation" / "final_consistent_metrics.json",
        "predictions": ROOT / "outputs" / "final_consistent_evaluation" / "test_predictions.npz",
        "metric_key": "test_metrics_at_validation_selected_threshold",
    },
    {
        "label": "EfficientNetB0 (frozen)",
        "metrics": ROOT / "outputs" / "efficientnetb0" / "metrics.json",
        "predictions": ROOT / "outputs" / "efficientnetb0" / "test_predictions.npz",
        "metric_key": "test_metrics_at_validation_selected_threshold",
    },
]

METRICS = [
    ("Accuracy", "accuracy"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Referable precision", "precision_referable"),
    ("Referable sensitivity", "sensitivity_recall_referable"),
    ("Non-referable specificity", "specificity_non_referable"),
    ("Referable F1", "f1_referable"),
    ("Macro-F1", "macro_f1"),
    ("ROC-AUC", "roc_auc"),
]


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required predictions file missing: {path}")
    with np.load(path, allow_pickle=True) as data:
        ids = data["id_code"].astype(str)
        y = data["y_true"].astype(int)
        p = data["probabilities"].astype(float)
    if not np.isfinite(p).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    return ids, y, p


def main():
    loaded = []
    reference_ids = reference_y = None

    for config in MODELS:
        raw_metrics = load_json(config["metrics"])
        if config["metric_key"] not in raw_metrics:
            raise KeyError(
                f"Expected key {config['metric_key']!r} in {config['metrics']}"
            )
        metrics = raw_metrics[config["metric_key"]]
        ids, y, probabilities = load_predictions(config["predictions"])

        if reference_ids is None:
            reference_ids, reference_y = ids, y
        else:
            if not np.array_equal(reference_ids, ids):
                raise ValueError(
                    f"Test image IDs/order do not match for {config['label']}"
                )
            if not np.array_equal(reference_y, y):
                raise ValueError(f"Test labels do not match for {config['label']}")

        loaded.append({
            **config, "metrics_data": metrics,
            "ids": ids, "y": y, "probabilities": probabilities,
        })

    if len(reference_y) != 550:
        print(f"Warning: expected 550 test images; found {len(reference_y)}")

    rows = []
    for display_name, key in METRICS:
        row = {"Metric": display_name}
        for item in loaded:
            row[f"{item['label']} (%)"] = float(item["metrics_data"][key]) * 100
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison_table.csv", index=False, float_format="%.4f")

    markdown = [
        "# Three-Model Comparison: Diabetic Retinopathy Detection",
        "",
        f"Test images: {len(reference_y)}",
        "All three prediction files were checked for identical test image IDs, order, and labels.",
        "",
        "| Metric | " + " | ".join(item["label"] for item in loaded) + " |",
        "|---|" + "---:|" * len(loaded),
    ]
    for _, row in table.iterrows():
        vals = " | ".join(f"{row[f'{item['label']} (%)']:.2f}%" for item in loaded)
        markdown.append(f"| {row['Metric']} | {vals} |")

    markdown.extend(["", "## Validation-selected thresholds", ""])
    for item in loaded:
        markdown.append(
            f"- **{item['label']}:** "
            f"{item['metrics_data']['threshold']:.2f}"
        )
    markdown.extend([
        "",
        "Thresholds were selected using validation data, not test data.",
        "The test set was previously examined during project development; these results are exploratory and do not constitute independent clinical validation.",
    ])
    (OUT / "comparison_table.md").write_text("\n".join(markdown), encoding="utf-8")

    x = np.arange(len(table))
    width = 0.24
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, item in enumerate(loaded):
        column = f"{item['label']} (%)"
        offset = (i - (len(loaded) - 1) / 2) * width
        bars = ax.bar(x + offset, table[column], width, label=item["label"])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)

    ax.set_ylabel("Score (%)")
    ax.set_title("Three-Model Comparison on the Same Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Metric"], rotation=25, ha="right")
    ax.set_ylim(70, 102)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Report-ready summary based on actual stored values.
    summary = [
        "## Results and Discussion",
        "",
        f"Three transfer-learning experiments were compared on the same {len(reference_y)} test images for binary classification of non-referable DR (grades 0-1) versus referable DR (grades 2-4). The prediction files were checked for identical test IDs, ordering, and labels.",
        "",
    ]
    for item in loaded:
        m = item["metrics_data"]
        summary.append(
            f"{item['label']} achieved accuracy {m['accuracy']*100:.2f}%, "
            f"balanced accuracy {m['balanced_accuracy']*100:.2f}%, "
            f"referable sensitivity {m['sensitivity_recall_referable']*100:.2f}%, "
            f"macro-F1 {m['macro_f1']*100:.2f}%, and ROC-AUC {m['roc_auc']*100:.2f}%. "
            f"Its validation-selected threshold was {m['threshold']:.2f}, with "
            f"{m['FN']} false negatives and {m['FP']} false positives."
        )
        summary.append("")
    summary.extend([
        "Metrics depending on a classification threshold reflect both model probabilities and each model's validation-selected operating point. The test set was previously examined during project development, so this comparison is exploratory and not independent clinical validation. External evaluation on an untouched dataset is required before drawing conclusions about clinical use or generalizability.",
    ])
    (OUT / "results_and_discussion.md").write_text("\n".join(summary), encoding="utf-8")

    print("\n=== THREE-MODEL COMPARISON COMPLETE ===")
    print(table.round(2).to_string(index=False))
    print(f"\nVerified identical test image IDs and labels: {len(reference_y)} images")
    print(f"Output folder: {OUT}")
    print("Created: comparison_table.csv, comparison_table.md, model_comparison.png, results_and_discussion.md")


if __name__ == "__main__":
    main()
