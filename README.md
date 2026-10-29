# Diabetic Retinopathy Detection Using Transfer Learning

Beginner-friendly B.Tech AI/ML project. This repository trains a binary classifier:
- `0`: No DR (APTOS diagnosis 0)
- `1`: DR (APTOS diagnosis 1, 2, 3, or 4)

Model: MobileNetV2 pretrained on ImageNet, then transfer-learned for the retinal image task.
Interface: Streamlit.

> **Important:** This is an educational research prototype, not a medical device or diagnostic service.

## 1. Dataset setup

This project uses the APTOS 2019 Blindness Detection training labels. Download the data from the official Kaggle competition page:
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

You must follow Kaggle's access and competition rules. Do not upload private credentials into the project.

Create this structure:

```text
diabetic_retinopathy_transfer_learning/
├── data/
│   ├── train.csv
│   └── train_images/
│       ├── <id_code>.png
│       └── ...
├── models/
├── outputs/
├── train.py
├── app.py
└── requirements.txt
```

`train.csv` must have `id_code` and `diagnosis` columns. Images may be PNG, JPG, or JPEG and should be named with the matching `id_code`.

The original APTOS labels are 0 through 4. This project converts them to binary labels: 0 stays No DR; 1-4 become DR.

## 2. Recommended easiest route: Google Colab

1. Upload `train.py` to a new Colab notebook or copy its code into a cell.
2. Set Colab runtime to GPU if available.
3. Put `train.csv` and `train_images/` somewhere accessible. For a large dataset, avoid loading every image into a NumPy array; the script streams images in batches.
4. If using Drive, mount it and change `ROOT` paths in the script to your Drive project directory, or copy the project/data into `/content` when storage permits.
5. Install dependencies if needed:

```python
!pip install -q pandas scikit-learn matplotlib
```

Colab commonly already includes TensorFlow. Check with:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

Run:

```python
!python train.py
```

Download/save the `models/best_model.keras` file and the `outputs/` folder before the runtime resets. Colab's temporary local storage does not persist after a runtime is deleted.

## 3. Run locally on Windows

Install Python, then open a terminal in this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

If PowerShell blocks activation, you can use Command Prompt and run `.venv\\Scripts\\activate.bat`, or run the environment's Python directly.

## 4. Launch the dashboard

After `models/best_model.keras` exists:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a retinal fundus image to get a model output. The dashboard also displays test-set metrics if training produced the `outputs/` files.

## 5. What the training script does

1. Reads APTOS labels and finds corresponding image files.
2. Converts grades into two classes.
3. Splits data into 70% train, 15% validation, and 15% test using stratification.
4. Loads and resizes images in batches rather than keeping all image pixels in RAM.
5. Trains a new classification head while MobileNetV2 is frozen.
6. Fine-tunes the last 30 layers with a small learning rate.
7. Uses class weights to reduce the effect of class imbalance.
8. Selects the best model by validation AUC.
9. Evaluates once on the held-out test split and saves metrics, report, confusion matrix, and split IDs.

The split is stratified, but if patient identifiers or multiple images per patient are available, use patient-level splitting to avoid leakage. Never use the held-out test set to tune model settings.

## 6. How to understand the metrics

- **Accuracy:** fraction of all test predictions that are correct.
- **Precision for DR:** among images predicted as DR, the fraction actually labelled DR.
- **Recall for DR (sensitivity):** among DR images, the fraction detected by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** how well the model ranks positive cases above negative cases across thresholds.
- **Confusion matrix:** counts correct and incorrect predictions for each class.

For this medical-image task, report recall, precision, F1, ROC-AUC, and the confusion matrix—not accuracy alone. The threshold of 0.5 is only a starting point; any threshold selection should be done using validation data, not test data.

## 7. Project deliverables

- Source code (`train.py`, `app.py`)
- Trained model (`models/best_model.keras`)
- Held-out test metrics (`outputs/metrics.json`)
- Classification report (`outputs/classification_report.txt`)
- Confusion matrix (`outputs/confusion_matrix.png`)
- Split files (`outputs/*_split.csv`)
- Screenshots of the dashboard and model results
- Report describing dataset, preprocessing, model, experimental setup, results, limitations, and future work

Training outputs are not included in the starter ZIP because they are generated only after you run the training script.

## 8. Limitations to state in your report

- This binary task combines mild through proliferative DR into one positive class, so it does not grade severity.
- Dataset labels and image quality may contain noise.
- Results on one dataset do not establish performance at other hospitals or with other cameras.
- Confidence scores may not be calibrated probabilities.
- This prototype must not be used for diagnosis or treatment decisions.
