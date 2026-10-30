
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "raw" / "MetroPT3(AirCompressor).csv"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# --------------------------------------------------
# 2. Define expected sensor columns
# --------------------------------------------------
sensor_columns = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current", "COMP", "DV_eletric",
    "Towers", "MPG", "LPS", "Pressure_switch",
    "Oil_level", "Caudal_impulses"
]

# --------------------------------------------------
# 3. Initialize data-quality counters
# --------------------------------------------------
missing_counts = {}
non_numeric_counts = {}
infinite_counts = {}
row_count = 0

# --------------------------------------------------
# 4. Read data in chunks
# --------------------------------------------------
chunk_size = 100_000

for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):

    row_count += len(chunk)

    for col in chunk.columns:
        missing_counts[col] = (
            missing_counts.get(col, 0)
            + int(chunk[col].isna().sum())
        )

    for col in sensor_columns:
        if col not in chunk.columns:
            continue

        # Convert a temporary copy to numeric.
        # Invalid text becomes NaN for counting.
        numeric = pd.to_numeric(chunk[col], errors="coerce")

        invalid_text = chunk[col].notna() & numeric.isna()

        non_numeric_counts[col] = (
            non_numeric_counts.get(col, 0)
            + int(invalid_text.sum())
        )

        infinite_counts[col] = (
            infinite_counts.get(col, 0)
            + int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        )

print("Total rows:", row_count)

print("\nMissing values by column:")
print(pd.Series(missing_counts).to_string())

print("\nNon-numeric sensor values:")
print(pd.Series(non_numeric_counts).to_string())

print("\nInfinite sensor values:")
print(pd.Series(infinite_counts).to_string())

# --------------------------------------------------
# 5. Build the data-quality report
# --------------------------------------------------
quality_report = {
    "total_rows": row_count,
    "missing_values": missing_counts,
    "non_numeric_sensor_values": non_numeric_counts,
    "infinite_sensor_values": infinite_counts
}

report_path = REPORT_DIR / "sensor_quality_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(quality_report, file, indent=4)

print("\nSaved quality report:", report_path)

# --------------------------------------------------
# 6. Inspect unusual values using IQR
# --------------------------------------------------
# This stage is an exploratory screen, not proof of failure.
# Read numeric sensor data in chunks and retain finite values.
sensor_values = {col: [] for col in sensor_columns}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=lambda col: col in sensor_columns,
    chunksize=chunk_size
):
    for col in sensor_columns:
        if col not in chunk.columns:
            continue

        values = pd.to_numeric(chunk[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        sensor_values[col].extend(values.tolist())

anomaly_summary = {}

for col, values in sensor_values.items():

    if not values:
        continue

    values = np.asarray(values, dtype=float)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    unusual = (values < lower_bound) | (values > upper_bound)

    anomaly_summary[col] = {
        "count": int(unusual.sum()),
        "percentage": float(unusual.mean() * 100),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "minimum": float(values.min()),
        "maximum": float(values.max())
    }

anomaly_df = pd.DataFrame(anomaly_summary).T

print("\nIQR unusual-value summary:")
print(anomaly_df.to_string())

anomaly_df.to_csv(REPORT_DIR / "iqr_anomaly_summary.csv")

# --------------------------------------------------
# 7. Plot selected sensor distributions
# --------------------------------------------------
plot_columns = [
    "TP2", "TP3", "Oil_temperature", "Motor_current"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flat, plot_columns):
    values = np.asarray(sensor_values[col], dtype=float)

    ax.hist(values, bins=60)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

plt.tight_layout()

plot_path = FIGURE_DIR / "sensor_distributions.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("\nSaved plots:", plot_path)
print("Step 4 inspection completed.")
