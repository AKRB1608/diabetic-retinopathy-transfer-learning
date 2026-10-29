from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import pennylane as qml

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "quantum_experiment" / "prepared"
OUT_DIR = ROOT / "quantum_experiment" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_QUBITS = 4
N_LAYERS = 2
MAX_TRAIN_SAMPLES = 1000
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 0.02

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
DEVICE = torch.device("cpu")
DTYPE = torch.float64

def load_split(name):
    path = DATA_DIR / f"{name}_quantum_inputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path, allow_pickle=False)
    return data["features"].astype(np.float64), data["labels"].astype(np.int64)

def metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def choose_f1_threshold(y_true, probs):
    # Validation-only threshold selection; test labels are never used here.
    thresholds = np.linspace(0.20, 0.80, 61)
    scored = [(f1_score(y_true, probs >= t, zero_division=0), t)
              for t in thresholds]
    best_f1 = max(x[0] for x in scored)
    # If tied, prefer the threshold nearest 0.5.
    candidates = [t for f1, t in scored if f1 == best_f1]
    return float(min(candidates, key=lambda t: abs(t - 0.5)))

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def main():
    X_train_all, y_train_all = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    # Select a reproducible stratified subset to keep simulator training practical.
    if len(y_train_all) > MAX_TRAIN_SAMPLES:
        X_train, _, y_train, _ = train_test_split(
            X_train_all, y_train_all,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=SEED,
            stratify=y_train_all,
        )
    else:
        X_train, y_train = X_train_all, y_train_all

    print("PennyLane:", qml.__version__)
    print("Torch:", torch.__version__)
    print("Simulator: default.qubit (CPU)")
    print(f"Training subset: {len(y_train)} / {len(y_train_all)}")
    print(f"Validation: {len(y_val)} | Test: {len(y_test)}")

    # Fair classical comparator: same 4 inputs and same training subset.
    print("\nTraining classical Logistic Regression...")
    classical = LogisticRegression(
        max_iter=2000, class_weight=None, random_state=SEED
    )
    classical.fit(X_train, y_train)
    val_classical = classical.predict_proba(X_val)[:, 1]
    test_classical = classical.predict_proba(X_test)[:, 1]
    classical_threshold = choose_f1_threshold(y_val, val_classical)

    classical_results = {
        "validation_at_0.5": metrics(y_val, val_classical, 0.5),
        "validation_selected_threshold": metrics(
            y_val, val_classical, classical_threshold
        ),
        "test_at_0.5": metrics(y_test, test_classical, 0.5),
        "test_at_validation_selected_threshold": metrics(
            y_test, test_classical, classical_threshold
        ),
        "selected_threshold_from_validation": classical_threshold,
    }
    save_json(OUT_DIR / "classical_logistic_regression.json", classical_results)

    # Four-qubit variational quantum circuit.
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for wire in range(N_QUBITS):
            qml.RY(inputs[wire], wires=wire)

        for layer in range(N_LAYERS):
            for wire in range(N_QUBITS):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for wire in range(N_QUBITS):
                qml.CNOT(wires=[wire, (wire + 1) % N_QUBITS])

        return qml.expval(qml.PauliZ(0))

    def predict_batch(X, weights):
        # Convert expectation z in [-1, 1] to probability (1-z)/2.
        outputs = []
        for row in X:
            x = torch.tensor(row, dtype=DTYPE, device=DEVICE)
            z = circuit(x, weights)
            outputs.append((1.0 - z) / 2.0)
        return torch.stack(outputs)

    weights = torch.nn.Parameter(
        0.05 * torch.randn(
            N_LAYERS, N_QUBITS, 2, dtype=DTYPE, device=DEVICE
        )
    )
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    loss_fn = torch.nn.BCELoss()

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    history = []
    start = time.time()

    print("\nTraining four-qubit VQC...")
    for epoch in range(1, EPOCHS + 1):
        order = np.random.permutation(len(y_train))
        weights.requires_grad_(True)
        batch_losses = []

        for start_idx in range(0, len(order), BATCH_SIZE):
            idx = order[start_idx:start_idx + BATCH_SIZE]
            xb = X_train[idx]
            yb = torch.tensor(y_train[idx], dtype=DTYPE, device=DEVICE)

            optimizer.zero_grad()
            probs = predict_batch(xb, weights)
            loss = loss_fn(probs.clamp(1e-7, 1 - 1e-7), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        with torch.no_grad():
            val_probs = predict_batch(X_val, weights).cpu().numpy()
        val_loss = float(
            loss_fn(
                torch.tensor(val_probs, dtype=DTYPE).clamp(1e-7, 1 - 1e-7),
                torch.tensor(y_val, dtype=DTYPE),
            ).item()
        )
        train_loss = float(np.mean(batch_losses))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"elapsed={(time.time()-start)/60:.1f} min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = weights.detach().clone()

    if best_weights is None:
        raise RuntimeError("VQC training did not produce a valid checkpoint.")

    weights_path = OUT_DIR / "vqc_weights.pt"
    torch.save(best_weights.cpu(), weights_path)
    save_json(OUT_DIR / "vqc_training_history.json", {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs": history,
        "qubits": N_QUBITS,
        "variational_layers": N_LAYERS,
        "training_subset_size": len(y_train),
        "full_training_size": len(y_train_all),
        "seed": SEED,
    })

    with torch.no_grad():
        val_vqc = predict_batch(X_val, best_weights).cpu().numpy()
        test_vqc = predict_batch(X_test, best_weights).cpu().numpy()

    vqc_threshold = choose_f1_threshold(y_val, val_vqc)
    vqc_results = {
        "validation_at_0.5": metrics(y_val, val_vqc, 0.5),
        "validation_selected_threshold": metrics(y_val, val_vqc, vqc_threshold),
        "test_at_0.5": metrics(y_test, test_vqc, 0.5),
        "test_at_validation_selected_threshold": metrics(
            y_test, test_vqc, vqc_threshold
        ),
        "selected_threshold_from_validation": vqc_threshold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
    }
    save_json(OUT_DIR / "vqc_results.json", vqc_results)

    comparison = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "validation_F1_at_0.5": classical_results["validation_at_0.5"]["f1"],
            "validation_ROC_AUC": classical_results["validation_at_0.5"]["roc_auc"],
            "test_accuracy_at_0.5": classical_results["test_at_0.5"]["accuracy"],
            "test_F1_at_0.5": classical_results["test_at_0.5"]["f1"],
            "test_ROC_AUC": classical_results["test_at_0.5"]["roc_auc"],
        },
        {
            "model": "4-qubit VQC",
            "validation_F1_at_0.5": vqc_results["validation_at_0.5"]["f1"],
            "validation_ROC_AUC": vqc_results["validation_at_0.5"]["roc_auc"],
            "test_accuracy_at_0.5": vqc_results["test_at_0.5"]["accuracy"],
            "test_F1_at_0.5": vqc_results["test_at_0.5"]["f1"],
            "test_ROC_AUC": vqc_results["test_at_0.5"]["roc_auc"],
        },
    ])
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print("\n=== TEST COMPARISON (threshold = 0.5) ===")
    print(comparison.to_string(index=False))
    print("\nVQC validation-selected threshold:", vqc_threshold)
    print("Saved results to:", OUT_DIR)
    print("NOTE: test results are for final comparison only, not model selection.")

if __name__ == "__main__":
    main()
