# Polymer Neural Network Testing
# Tests both the Phase-Space and Coordinate Networks

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


# ============================================================
# Helper Function
# ============================================================

def test_model(model_path, test_data_path, test_answer_path, name):
    """
    Load a trained model and evaluate it using its
    corresponding test dataset.
    """

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load test inputs
    TestInputDf = pd.read_csv(test_data_path)
    TestInput = TestInputDf.to_numpy(dtype=np.float32)

    # Load test answers
    TestAnsDf = pd.read_csv(test_answer_path)
    TestAns = (
        TestAnsDf.to_numpy()
        .squeeze()
        .astype(np.int64)
    )

    print(f"Test samples: {len(TestInput)}")

    # Evaluate
    loss, accuracy = model.evaluate(
        TestInput,
        TestAns,
        verbose=1
    )

    print(f"Loss: {loss:.6f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return loss, accuracy


# ============================================================
# Test Phase-Space Network
# ============================================================

phase_loss, phase_accuracy = test_model(
    OUTPUT_DIR / "PolymerNeuralNetwork.keras",
    DATA_DIR / "TestDataL40W0.75.csv",
    DATA_DIR / "TestAnsL40W0.75.csv",
    "Phase-Space Neural Network"
)


# ============================================================
# Test Coordinate Network
# ============================================================

coordinate_loss, coordinate_accuracy = test_model(
    OUTPUT_DIR / "PolymerCordsNeuralNetwork.keras",
    DATA_DIR / "TestDataCordsL40W0.75.csv",
    DATA_DIR / "TestAnsCordsL40W0.75.csv",
    "Coordinate Neural Network"
)


# ============================================================
# Final Comparison
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"Phase-Space Network: "
    f"{phase_accuracy * 100:.2f}%"
)

print(
    f"Coordinate Network: "
    f"{coordinate_accuracy * 100:.2f}%"
)

difference = (
    phase_accuracy - coordinate_accuracy
) * 100

print(
    f"Difference: {difference:+.2f} percentage points"
)