from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
THRESHOLD_PATH = ROOT / "outputs" / "referable_threshold.json"
TEST_METRICS_PATH = ROOT / "outputs" / "test_metrics_recomputed.json"
VALIDATION_METRICS_PATH = ROOT / "outputs" / "validation_metrics_recomputed.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="RetinaAI | Referable DR + Grad-CAM", page_icon="👁️", layout="wide")

st.markdown("""
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
""", unsafe_allow_html=True)

header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("👁️ RetinaAI")
    st.markdown("**Referable Diabetic Retinopathy Detection with Grad-CAM**")
with header_right:
    st.caption("RESEARCH PROTOTYPE")

with st.expander("Class mapping and important limitations", expanded=False):
    st.markdown("- **Grades 0–1:** Non-referable DR (No DR / Mild DR)\n- **Grades 2–4:** Referable DR (Moderate / Severe / Proliferative DR)")
    st.warning("This application is a research prototype, not a medical diagnosis or validated clinical referral recommendation. Do not use it to delay or replace assessment by a qualified eye-care professional.")

@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def build_gradcam_model(model_path: str):
    """Expose MobileNetV2's final feature map and the model output."""
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
        raise ValueError("Could not find MobileNetV2's 'out_relu' layer. The saved model architecture may differ from this app's expected architecture.")
    feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.get_layer("out_relu").output, name="gradcam_feature_extractor")
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
    # Model contains Rescaling(1/127.5, offset=-1); provide RGB pixels in 0–255.
    resized = pil_image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
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
    heatmap = tf.image.resize(heatmap[..., None], IMAGE_SIZE, method="bilinear").numpy().squeeze()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    original_array = np.asarray(original, dtype=np.float32) / 255.0
    color_map = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.55 * original_array + 0.45 * color_map, 0.0, 1.0)
    overlay = (overlay * 255).astype(np.uint8)
    return original, heatmap, overlay

def load_saved_threshold() -> float:
    """Use the threshold selected on validation data, never the test set."""
    if THRESHOLD_PATH.exists():
        try:
            saved = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
            value = float(saved.get("threshold", 0.71))
            if np.isfinite(value):
                return float(np.clip(value, 0.05, 0.95))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return 0.71

def show_saved_evaluation(path: Path, title: str, sample_key: str, show_matrix: bool = True):
    """Display the recomputed evaluation JSON, rather than older NPZ predictions."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load {title} from {path.name}: {exc}")
        return
    required = [sample_key, "threshold", "accuracy", "balanced_accuracy", "referable_precision", "referable_recall", "non_referable_specificity", "referable_f1", "macro_f1", "roc_auc", "confusion_matrix"]
    missing = [field for field in required if field not in data]
    if missing:
        st.warning(f"{title}: missing fields in {path.name}: {', '.join(missing)}")
        return
    try:
        matrix = np.asarray(data["confusion_matrix"], dtype=int)
    except (ValueError, TypeError):
        st.warning(f"{title}: confusion matrix is not valid numeric data.")
        return
    if matrix.shape != (2, 2):
        st.warning(f"{title}: expected a 2 × 2 confusion matrix.")
        return
    st.subheader(title)
    st.caption(f"{data[sample_key]} images · threshold {float(data['threshold']):.2f}")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{float(data['accuracy']):.2%}")
    cols[1].metric("Balanced accuracy", f"{float(data['balanced_accuracy']):.2%}")
    cols[2].metric("Sensitivity", f"{float(data['referable_recall']):.2%}")
    cols[3].metric("Specificity", f"{float(data['non_referable_specificity']):.2%}")
    cols2 = st.columns(4)
    cols2[0].metric("Precision", f"{float(data['referable_precision']):.2%}")
    cols2[1].metric("Referable F1", f"{float(data['referable_f1']):.2%}")
    cols2[2].metric("Macro-F1", f"{float(data['macro_f1']):.2%}")
    cols2[3].metric("ROC-AUC", f"{float(data['roc_auc']):.2%}")
    if not show_matrix:
        return
    st.markdown("**Confusion matrix**")
    st.caption("Rows = actual class · Columns = predicted class")
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Non-referable", "Referable"])
    ax.set_yticks([0, 1], labels=["Non-referable", "Referable"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title, fontsize=11, color="#153b50")
    cutoff = matrix.max() * 0.55
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > cutoff else "#153b50", fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

if not MODEL_PATH.exists():
    st.error(f"Trained referable model not found: {MODEL_PATH}")
    st.stop()
try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

saved_threshold = load_saved_threshold()
st.header("Image analysis")
upload_col, threshold_col = st.columns([1.15, 1], gap="large")
with upload_col:
    uploaded = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"], help="Use a retinal fundus photograph. Images are resized to 224 × 224.")
with threshold_col:
    prediction_threshold = st.slider("Prediction decision threshold", min_value=0.05, max_value=0.95, value=float(round(saved_threshold, 2)), step=0.01, help="This slider changes only the current uploaded-image decision. It does not retrain the model or alter the fixed evaluation.")
    st.caption(f"Default threshold: {saved_threshold:.2f}, selected using validation data. The score is not guaranteed to be a calibrated probability.")

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
        st.caption(f"Decision rule: referable DR when score ≥ {prediction_threshold:.2f}. A model prediction can be wrong and is not a diagnosis.")
        st.header("Grad-CAM visual explanation")
        st.caption("Grad-CAM highlights image regions that influenced the model's referable-DR score. It does not confirm that a lesion is present.")
        with st.spinner("Generating Grad-CAM heatmap..."):
            grad_model = build_gradcam_model(str(MODEL_PATH))
            original, heatmap, overlay = make_gradcam_overlay(grad_model, image)
        image_col, heatmap_col, overlay_col = st.columns(3, gap="medium")
        with image_col:
            st.image(original, caption="Original image", use_container_width=True)
        with heatmap_col:
            st.image(heatmap, caption="Grad-CAM heatmap", clamp=True, use_container_width=True)
        with overlay_col:
            st.image(overlay, caption="Heatmap overlay", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not process this image or generate Grad-CAM: {exc}")
else:
    st.info("Upload a retinal fundus image to see the prediction and Grad-CAM.")

st.divider()
st.header("Model evaluation")
st.caption("Binary referable diabetic retinopathy classification · Research use only")
show_saved_evaluation(TEST_METRICS_PATH, "Held-out test results", "test_samples", show_matrix=True)
with st.expander("Validation results", expanded=False):
    show_saved_evaluation(VALIDATION_METRICS_PATH, "Validation results", "validation_samples", show_matrix=True)
st.caption("The uploaded-image threshold slider affects individual predictions only. Evaluation results are loaded from the recomputed JSON files. This research prototype is not a clinically validated diagnostic system.")
