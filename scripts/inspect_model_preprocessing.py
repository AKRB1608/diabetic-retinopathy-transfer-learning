from pathlib import Path
import tensorflow as tf

project = Path(r"D:\Projects\diabetic-retinopathy-starter\diabetic_retinopathy_transfer_learning")
model_path = project / "models" / "best_referable_model.keras"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)

print("\n=== MODEL INPUT ===")
print(model.input_shape)

print("\n=== TOP-LEVEL LAYERS ===")
for i, layer in enumerate(model.layers):
    print(f"{i:02d}. {layer.name}: {layer.__class__.__name__}")

print("\n=== RESCALING LAYERS ===")
found = False
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Rescaling):
        found = True
        print(
            f"{layer.name}: scale={layer.scale}, "
            f"offset={layer.offset}"
        )
    if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            if isinstance(sublayer, tf.keras.layers.Rescaling):
                found = True
                print(
                    f"{layer.name}/{sublayer.name}: "
                    f"scale={sublayer.scale}, offset={sublayer.offset}"
                )

if not found:
    print("No Rescaling layer found at the inspected levels.")

print("\n=== END OF INSPECTION ===")
