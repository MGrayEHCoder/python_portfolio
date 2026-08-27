# Polymer Neural Network Training
# Phase-Space / Property-Based Network

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Load Training Data
# ============================================================

TrainingInputDf = pd.read_csv(
    DATA_DIR / "TrainingDataL40W0.75.csv"
)

TrainingAnswerDf = pd.read_csv(
    DATA_DIR / "TrainingAnsL40W0.75.csv"
)

TrainingInput = TrainingInputDf.to_numpy(dtype=np.float32)

TrainingAnswer = (
    TrainingAnswerDf.to_numpy()
    .squeeze()
    .astype(np.int64)
)


# ============================================================
# Check Labels
# ============================================================

unique_classes = np.unique(TrainingAnswer)
num_classes = len(unique_classes)

print("Classes:", unique_classes)
print("Number of classes:", num_classes)
print("Training samples:", len(TrainingInput))


if unique_classes.min() < 0 or unique_classes.max() >= num_classes:
    raise ValueError(
        "Training labels must be zero-based integers "
        "from 0 to num_classes - 1."
    )


# ============================================================
# Define Model
# ============================================================

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(
        shape=(TrainingInput.shape[1],)
    ),

    tf.keras.layers.Dense(
        40,
        activation="swish"
    ),

    tf.keras.layers.Dense(
        50,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        25,
        activation="elu"
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Train Model
# ============================================================

history = model.fit(
    TrainingInput,
    TrainingAnswer,
    epochs=5,
    shuffle=True
)


# ============================================================
# Print Training Performance
# ============================================================

final_accuracy = history.history["accuracy"][-1]
final_loss = history.history["loss"][-1]

print(f"Final training accuracy: {final_accuracy:.4f}")
print(f"Final training loss: {final_loss:.4f}")


# ============================================================
# Save Model
# ============================================================

model_path = OUTPUT_DIR / "PolymerNeuralNetwork.keras"

model.save(model_path)

print(f"Model saved to: {model_path}")