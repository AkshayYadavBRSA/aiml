
from pathlib import Path
import pandas as pd

# Locate the project and raw data
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "MetroPT3(AirCompressor).csv"

# Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please place the CSV inside data/raw/"
    )

print("Dataset found successfully!")
print("File size:", round(DATA_PATH.stat().st_size / (1024 ** 2), 2), "MB")

# Read a small sample first
df = pd.read_csv(DATA_PATH, nrows=10)

print("\nFirst 10 rows:")
print(df.to_string(index=False))

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)
