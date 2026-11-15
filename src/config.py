from pathlib import Path

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

TARGET = "Machine failure"
CAT_COL = "Type"
ID_COLS = ["UDI", "Product ID"]
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUM_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

PHYSICS_PAIRS = [
    ("Torque [Nm]", "Rotational speed [rpm]"),
    ("Tool wear [min]", "Torque [Nm]"),
    ("Air temperature [K]", "Process temperature [K]"),
]
