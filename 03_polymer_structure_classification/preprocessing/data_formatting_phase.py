import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Load original simulation data
df = pd.read_csv(DATA_DIR / "ListOfStructuresL40W0.75.csv")

# Drop coordinate data
for axis in ["x", "y", "z"]:
    for j in range(40):
        df = df.drop(f"{axis}{j}", axis=1)

# Drop simulation data and indexes
columns_to_drop = [
    "Step",
    "Bin",
    "Unnamed: 0",
    "Unnamed: 0.3",
    "Unnamed: 0.2",
    "Unnamed: 0.1",
    "Replica",
    "Temp",
    "Width",
    "NMon",
    "BendingStrength",
    "LJFrac",
    "Energy",
]

df = df.drop(columns=columns_to_drop)

# Shuffle data
Sdf = df.sample(frac=1, random_state=24).reset_index(drop=True)

# Separate labels
Structures = Sdf["StructureType"]

# Determine all classes present in this dataset
classes = sorted(Structures.unique())

# Map them to consecutive zero-based labels
class_mapping = {
    original: new
    for new, original in enumerate(classes)
}

Structures = Structures.map(class_mapping)

print("Original classes:", classes)
print("Class mapping:", class_mapping)
print("Mapped classes:", sorted(Structures.unique()))

# Remove labels from input data
Sdf = Sdf.drop(
    columns=["CategoryNumber", "CatagoryNumber", "StructureType"]
)

# Split data BEFORE normalization
n_rows = int(len(Sdf) * 0.5)

TrainingRaw = Sdf.iloc[:n_rows].copy()
TestRaw = Sdf.iloc[n_rows:].copy()

TrainingAns = Structures.iloc[:n_rows].copy()
TestAns = Structures.iloc[n_rows:].copy()

# Normalize using TRAINING data only
train_min = TrainingRaw.min()
train_max = TrainingRaw.max()

TrainingData = (TrainingRaw - train_min) / (train_max - train_min)
TestData = (TestRaw - train_min) / (train_max - train_min)

# Save processed data
TrainingData.to_csv(
    DATA_DIR / "TrainingDataL40W0.75.csv",
    index=False
)

TrainingAns.to_csv(
    DATA_DIR / "TrainingAnsL40W0.75.csv",
    index=False
)

TestData.to_csv(
    DATA_DIR / "TestDataL40W0.75.csv",
    index=False
)

TestAns.to_csv(
    DATA_DIR / "TestAnsL40W0.75.csv",
    index=False
)

print(f"Training samples: {len(TrainingData)}")
print(f"Test samples: {len(TestData)}")
print(f"Classes: {Structures.unique()}")