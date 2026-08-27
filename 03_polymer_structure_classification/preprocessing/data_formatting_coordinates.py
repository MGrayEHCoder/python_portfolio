import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Load original simulation data
df = pd.read_csv(DATA_DIR / "ListOfStructuresL40W0.75.csv")

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
    "LJTot",
    "LJNear",
    "LJFar",
    "Contacts",
    "NearCont",
    "FarCont",
    "AvgAngles",
    "StdDevAngles",
    "MaxDiffAngles",
    "EndToEndLength",
]

df = df.drop(columns=columns_to_drop)

# Shuffle data
Sdf = df.sample(frac=1, random_state=24).reset_index(drop=True)

# Separate labels
Structures = Sdf["StructureType"]

# Remove labels
Sdf = Sdf.drop(
    columns=["CategoryNumber", "CatagoryNumber", "StructureType"]
)

# Convert coordinate data into groups of XYZ
coords = Sdf.to_numpy().reshape(len(Sdf), -1, 3)

# Translate each polymer so its minimum coordinate is at the origin
row_min = coords.min(axis=1, keepdims=True)
shifted = coords - row_min

# Convert back to DataFrame
ShiftedDf = pd.DataFrame(
    shifted.reshape(len(Sdf), -1),
    columns=Sdf.columns
)

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


# Split BEFORE normalization
n_rows = int(len(Sdf) * 0.5)

TrainingRaw = ShiftedDf.iloc[:n_rows].copy()
TestRaw = ShiftedDf.iloc[n_rows:].copy()

TrainingAns = Structures.iloc[:n_rows].copy()
TestAns = Structures.iloc[n_rows:].copy()

# Normalize using TRAINING data only
train_min = TrainingRaw.min()
train_max = TrainingRaw.max()

TrainingData = (TrainingRaw - train_min) / (train_max - train_min)
TestData = (TestRaw - train_min) / (train_max - train_min)

# Save processed data
TrainingData.to_csv(
    DATA_DIR / "TrainingCordsDataL40W0.75.csv",
    index=False
)

TrainingAns.to_csv(
    DATA_DIR / "TrainingCordsAnsL40W0.75.csv",
    index=False
)

TestData.to_csv(
    DATA_DIR / "TestDataCordsL40W0.75.csv",
    index=False
)

TestAns.to_csv(
    DATA_DIR / "TestAnsCordsL40W0.75.csv",
    index=False
)

print(f"Training samples: {len(TrainingData)}")
print(f"Test samples: {len(TestData)}")
print(f"Classes: {Structures.unique()}")