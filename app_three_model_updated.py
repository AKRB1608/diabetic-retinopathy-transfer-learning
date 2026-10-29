from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)

MODEL_CONFIG = {
    "Fine-tuned MobileNetV2 (recommended)": {
        "path": ROOT / "models" / "best_referable_model.keras",
        "threshold_path": ROOT / "outputs" / "referable_threshold.json",
        "metrics_path": ROOT / "outputs" / "final_consistent_evaluation" / "final_consistent_metrics.json",
        "prediction_path": ROOT / "outputs" / "final_consistent_evaluation" / "test_predictions.npz",
        "threshold_fallback": 0.51,
        "gradcam": True,
    },
    "Frozen MobileNetV2 (baseline)": {
        "path": ROOT / "models" / "best_frozen_baseline.keras",
        "threshold_path": ROOT / "outputs" / "frozen_baseline" / "threshold.json",
        "metrics_path": ROOT / "outputs" / "frozen_baseline" / "metrics.json",
        "prediction_path": ROOT / "outputs" / "frozen_baseline" / "test_predictions.npz",
        "threshold_fallback": 0.37,
        "gradcam": False,
    },
    "EfficientNetB0 (frozen backbone)": {
        "path": ROOT / "models" / "best_efficientnetb0.keras",
        "threshold_path": ROOT / "outputs" / "efficientnetb0" / "threshold.json",
        "metrics_path": ROOT / "outputs" / "efficientnetb0" / "metrics.json",
        "prediction_path": ROOT / "outputs" / "efficientnetb0" / "test_predictions.npz",
        "threshold_fallback": 0.50,
        "gradcam": False,
    },
}

st.set_page_config(
    page_title="RetinaAI | DR Research Prototype",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background:#fff; color:#172b3a; }
    .block-container { max-width:1180px; padding-top:1.35rem; padding-bottom:2rem; }
    h1,h2,h3 { color:#153b50 !important; letter-spacing:-.02em; }
    div[data-testid="stMetric"] { background:#f5f8fb; border:1px solid #dce6ee; border-radius:12px; padding:12px 14px; }
    div[data-testid="stMetricLabel"] { color:#526b7b; }
    div[data-testid="stMetricValue"] { color:#153b50; }
    div[data-testid="stFileUploader"] { background:#f8fafc; border:1px dashed #9aafbf; border-radius:12px; padding:10px; }
    div[data-testid="stExpander"] { border:1px solid #dce6ee; border-radius:10px; }
    hr { border-color:#dce6ee; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👁️ RetinaAI")
st.markdown("**Diabetic Retinopathy Detection Using Transfer Learning and Grad-CAM**")
st.caption("RESEARCH PROTOTYPE")

with st.expander("Class mapping and important limitations", expanded=False):
    st.markdown(
        "- **Grades 0–1:** Non-referable DR (No DR / Mild DR)\n"
        "- **Grades 2–4:** Referable DR (Moderate / Severe / Proliferative DR)"
    )
    st.warning(
        "This application is a research prototype, not a medical diagnosis or "
        "validated clinical referral recommendation. Do not use it to delay or "
        "replace assessment by a qualified eye-care professional."
    )


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def build_mobilenet_gradcam_model(model_path: str):
    """Build Grad-CAM model for the project's nested MobileNetV2 architecture."""
    model = tf.keras.models.load_model(model_path)
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer("out_relu")
                base = layer
                break
            except ValueError:
                continue

    if base is None:
        raise ValueError(
            "Could not find MobileNetV2's 'out_relu' layer. "
            "Grad-CAM architecture does not match this saved model."
        )

    feature_extractor = tf.keras.Model(
        inputs=base.input,
        outputs=base.get_layer("out_relu").output,
        name="gradcam_feature_extractor",
    )
    inputs = model.inputs[0]
    x = inputs
    conv_outputs = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is base:
            x = feature_extractor(x)
            conv_outputs = x
        else:
            x = layer(x)
    if conv_outputs is None:
        raise ValueError("Could not connect the Grad-CAM feature layer.")
    return tf.keras.Model(inputs=inputs, outputs=[conv_outputs, x])


def prepare_image(pil_image: Image.Image):
    resized = pil_image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    # The saved models contain their own model-specific preprocessing.
    return resized, tf.convert_to_tensor(array[None, ...], dtype=tf.float32)


def predict_score(model, pil_image: Image.Image) -> float:
    _, batch = prepare_image(pil_image)
    output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if output.size == 0 or not np.isfinite(output[0]):
        raise ValueError("The model returned an invalid prediction.")
    return float(np.clip(output[0], 0.0, 1.0))


def make_gradcam_overlay(grad_model, pil_image: Image.Image):
    original, batch = prepare_image(pil_image)
    with tf.GradientTape() as tape:
        conv_features, prediction = grad_model(batch, training=False)
        score = prediction[:, 0]
    gradients = tape.gradient(score, conv_features)
    if gradients is None:
        raise ValueError("Gradients were unavailable for Grad-CAM.")
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))[0]
    features = conv_features[0]
    heatmap = tf.reduce_sum(features * channel_weights[None, None, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(
        heatmap[..., None], IMAGE_SIZE, method="bilinear"
    ).numpy().squeeze()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    original_array = np.asarray(original, dtype=np.float32) / 255.0
    color_map = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.55 * original_array + 0.45 * color_map, 0.0, 1.0)
    return original, heatmap, (overlay * 255).astype(np.uint8)


def load_threshold(config):
    path = config["threshold_path"]
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            threshold = float(saved.get("threshold", config["threshold_fallback"]))
            if np.isfinite(threshold):
                return float(np.clip(threshold, 0.05, 0.95))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return float(config["threshold_fallback"])


def unwrap_metrics(data):
    """Handle both direct metrics JSON and wrapper JSON formats."""
    if not isinstance(data, dict):
        return None
    for key in (
        "test_metrics_at_validation_selected_threshold",
        "validation_metrics_at_selected_threshold",
        "metrics",
    ):
        if isinstance(data.get(key), dict):
            return data[key]
    return data


def load_evaluation_metrics(config):
    path = config["metrics_path"]
    if not path.exists():
        return None, f"Metrics file not found: {path.relative_to(ROOT)}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"Could not read metrics file: {exc}"

    metrics = unwrap_metrics(raw)
    if not isinstance(metrics, dict):
        return None, "Metrics file has an unsupported format."

    # Normalize names used by the project's different evaluation scripts.
    aliases = {
        "accuracy": ("accuracy",),
        "balanced_accuracy": ("balanced_accuracy",),
        "precision": ("precision_referable", "referable_precision", "precision"),
        "sensitivity": (
            "sensitivity_recall_referable", "referable_recall", "sensitivity", "recall"
        ),
        "specificity": (
            "specificity_non_referable", "non_referable_specificity", "specificity"
        ),
        "f1": ("f1_referable", "referable_f1", "f1"),
        "macro_f1": ("macro_f1",),
        "roc_auc": ("roc_auc",),
        "threshold": ("threshold", "validation_selected_threshold"),
        "confusion_matrix": ("confusion_matrix_labels_0_1", "confusion_matrix"),
        "samples": ("samples", "test_samples", "validation_samples"),
    }
    normalized = {}
    for target, candidates in aliases.items():
        for key in candidates:
            if key in metrics:
                normalized[target] = metrics[key]
                break
    return normalized, None


def show_evaluation(config, title):
    metrics, error = load_evaluation_metrics(config)
    st.subheader(title)
    if error:
        st.info(f"{error}. Run the model's evaluation script to create it.")
        return
    if not metrics or "accuracy" not in metrics:
        st.info("This metrics file does not contain the expected evaluation fields.")
        return

    def pct(key):
        try:
            return f"{float(metrics[key]):.2%}"
        except (KeyError, ValueError, TypeError):
            return "N/A"

    st.caption(
        "Saved evaluation results. Threshold-dependent metrics use the threshold "
        "selected on validation data."
    )
    cols = st.columns(4)
    cols[0].metric("Accuracy", pct("accuracy"))
    cols[1].metric("Balanced accuracy", pct("balanced_accuracy"))
    cols[2].metric("Sensitivity", pct("sensitivity"))
    cols[3].metric("Specificity", pct("specificity"))
    cols2 = st.columns(4)
    cols2[0].metric("Precision", pct("precision"))
    cols2[1].metric("Referable F1", pct("f1"))
    cols2[2].metric("Macro-F1", pct("macro_f1"))
    cols2[3].metric("ROC-AUC", pct("roc_auc"))

    matrix = metrics.get("confusion_matrix")
    if matrix is not None:
        try:
            matrix = np.asarray(matrix, dtype=int)
            if matrix.shape == (2, 2):
                st.markdown("**Confusion matrix**")
                st.caption("Rows = actual class; columns = predicted class.")
                fig, ax = plt.subplots(figsize=(4.8, 3.5))
                ax.imshow(matrix, cmap="Blues")
                ax.set_xticks([0, 1], labels=["Non-referable", "Referable"])
                ax.set_yticks([0, 1], labels=["Non-referable", "Referable"])
                ax.set_xlabel("Predicted class")
                ax.set_ylabel("Actual class")
                ax.set_title(title, fontsize=11)
                cutoff = matrix.max() * 0.55 if matrix.size else 0
                for row in range(2):
                    for col in range(2):
                        ax.text(
                            col, row, str(matrix[row, col]),
                            ha="center", va="center",
                            color="white" if matrix[row, col] > cutoff else "#153b50",
                            fontsize=12, fontweight="bold",
                        )
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        except (ValueError, TypeError):
            st.warning("The saved confusion matrix is not valid numeric data.")


st.header("Image analysis")
selected_name = st.selectbox(
    "Choose a model",
    list(MODEL_CONFIG.keys()),
    index=0,
    help="Each model has its own checkpoint and validation-selected threshold.",
)
config = MODEL_CONFIG[selected_name]
model_path = config["path"]

if not model_path.exists():
    st.error(f"Model file not found: {model_path.relative_to(ROOT)}")
    st.info("Check that the model has been trained and saved at this path.")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()

saved_threshold = load_threshold(config)
upload_col, threshold_col = st.columns([1.15, 1], gap="large")
with upload_col:
    uploaded = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"],
        help="Use a retinal fundus photograph. Images are resized to 224 × 224.",
        key="retinal_upload",
    )
with threshold_col:
    prediction_threshold = st.slider(
        "Prediction decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(round(saved_threshold, 2)),
        step=0.01,
        help="Changes only the uploaded-image decision; it does not retrain the model.",
        key=f"threshold_{selected_name}",
    )
    st.caption(
        f"Default threshold: {saved_threshold:.2f}. The saved threshold is intended "
        "to come from validation data. Scores are not guaranteed to be calibrated probabilities."
    )

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        score = predict_score(model, image)
        st.divider()
        st.header("Prediction")
        result_col, score_col = st.columns([1.2, 1])
        with result_col:
            if score >= prediction_threshold:
                st.error("Referable DR predicted (grades 2–4)")
            else:
                st.success("Non-referable DR predicted (grades 0–1)")
        with score_col:
            st.metric("Model referable-DR score", f"{score:.1%}")
        st.caption(
            f"Decision rule: referable DR when score ≥ {prediction_threshold:.2f}. "
            "Predictions can be wrong and are not a diagnosis."
        )

        if config["gradcam"]:
            st.header("Grad-CAM visual explanation")
            st.caption(
                "Grad-CAM highlights image regions that influenced the model score. "
                "It does not confirm that a lesion is present."
            )
            with st.spinner("Generating Grad-CAM heatmap..."):
                grad_model = build_mobilenet_gradcam_model(str(model_path))
                original, heatmap, overlay = make_gradcam_overlay(grad_model, image)
            image_col, heatmap_col, overlay_col = st.columns(3, gap="medium")
            with image_col:
                st.image(original, caption="Original image", use_container_width=True)
            with heatmap_col:
                st.image(heatmap, caption="Grad-CAM heatmap", clamp=True, use_container_width=True)
            with overlay_col:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.info(
                "Grad-CAM is currently enabled only for the fine-tuned MobileNetV2 "
                "because that model's Grad-CAM layer connection is implemented and "
                "must be verified separately for other architectures."
            )
    except Exception as exc:
        st.error(f"Could not process this image or generate Grad-CAM: {exc}")
else:
    st.info("Upload a retinal fundus image to see a prediction.")

st.divider()
st.header("Model evaluation")
st.caption("Binary classification: grades 0–1 = non-referable; grades 2–4 = referable.")
show_evaluation(config, f"{selected_name} — test results")

with st.expander("Validation results", expanded=False):
    # For EfficientNetB0, validation metrics are nested in its metrics.json.
    # For baseline, metrics.json includes validation and test metric objects.
    metrics_path = config["metrics_path"]
    if metrics_path.exists():
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            val_data = raw.get("validation_metrics_at_selected_threshold")
            if val_data is None and "validation_metrics" in raw:
                val_data = raw["validation_metrics"]
            if isinstance(val_data, dict):
                st.json(val_data)
            else:
                st.info(
                    "Validation metrics are not stored as a separate displayable "
                    "object in this file. The decision threshold was saved separately."
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Could not load validation metrics: {exc}")

st.caption(
    "This research prototype is not clinically validated and must not replace assessment "
    "by a qualified eye-care professional. Test-set performance is exploratory because "
    "the test set has been examined during project development."
)
