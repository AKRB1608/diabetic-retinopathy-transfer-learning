from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "raw" / "train_images"
SPLIT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "best_referable_model.keras"
OUTPUT_DIR = ROOT / "quantum_experiment" / "features"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

np.random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def locate_image(image_id):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.is_file():
            return path
    return None

def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(
        raw, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    # Keep pixels in 0-255 range: the saved model contains Rescaling.
    image = tf.cast(image, tf.float32)
    return image, label

def main():
    for required in (MODEL_PATH, IMAGE_DIR, SPLIT_DIR):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Extract learned features before dropout and the final prediction layer.
    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("global_average_pooling2d").output,
        name="retinal_feature_extractor",
    )
    print("Feature output:", feature_model.output_shape)
    print("Model:", MODEL_PATH)

    for split in ("train", "validation", "test"):
        csv_path = SPLIT_DIR / f"{split}_split.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"id_code", "diagnosis", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{csv_path.name} needs {required_cols}; "
                f"found {list(df.columns)}"
            )

        df["id_code"] = df["id_code"].astype(str)
        df["diagnosis"] = df["diagnosis"].astype(int)
        df["label"] = df["label"].astype(int)

        # Verify the intended binary mapping: diagnosis >= 2 is referable.
        expected = (df["diagnosis"] >= 2).astype(int)
        if not np.array_equal(expected.to_numpy(), df["label"].to_numpy()):
            raise ValueError(
                f"{split}: label column does not match diagnosis >= 2."
            )

        paths = [locate_image(image_id) for image_id in df["id_code"]]
        missing = [
            image_id for image_id, path in zip(df["id_code"], paths)
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"{split}: {len(missing)} images missing. "
                f"Examples: {missing[:5]}"
            )

        ds = tf.data.Dataset.from_tensor_slices(
            ([str(p) for p in paths], df["label"].to_numpy(dtype=np.int32))
        )
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        feature_batches = []
        for batch_images, _ in ds:
            batch_features = feature_model(batch_images, training=False)
            feature_batches.append(batch_features.numpy())

        features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        labels = df["label"].to_numpy(dtype=np.int32)

        if features.shape != (len(df), 1280):
            raise ValueError(f"Unexpected {split} feature shape: {features.shape}")

        output_path = OUTPUT_DIR / f"{split}_features.npz"
        np.savez_compressed(
            output_path,
            features=features,
            labels=labels,
            diagnosis=df["diagnosis"].to_numpy(dtype=np.int32),
            id_codes=df["id_code"].to_numpy(dtype=str),
        )

        print(
            f"{split:10s} | samples={len(df):4d} | "
            f"features={features.shape} | "
            f"class_counts={dict(zip(*np.unique(labels, return_counts=True)))}"
        )
        print(f"Saved: {output_path}")

    print("\nFEATURE EXTRACTION COMPLETED")
    print("Original model and split CSVs were not modified.")
    print("Next: fit feature scaling/PCA on training features only.")

if __name__ == "__main__":
    main()
