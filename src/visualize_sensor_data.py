
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "MetroPT3(AirCompressor).csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# --------------------------------------------------
# 2. Define columns
# --------------------------------------------------
sensor_columns = [
    "TP2",
    "TP3",
    "H1",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

required_columns = ["timestamp"] + sensor_columns

# --------------------------------------------------
# 3. Read data in chunks
# --------------------------------------------------
# Plotting does not require every row.
# We retain every 100th row for the long-term overview.
overview_parts = []
chunk_size = 100_000
sample_every = 100
row_offset = 0

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=required_columns,
    chunksize=chunk_size,
):
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"], errors="coerce"
    )

    for col in sensor_columns:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        )

    # Keep a consistent sample across chunk boundaries
    positions = [
        i for i in range(len(chunk))
        if (row_offset + i) % sample_every == 0
    ]

    overview_parts.append(chunk.iloc[positions].copy())
    row_offset += len(chunk)

overview = pd.concat(overview_parts, ignore_index=True)

overview = (
    overview.dropna(subset=["timestamp"])
    .sort_values("timestamp")
)

print("Rows in original dataset:", row_offset)
print("Rows used for overview plots:", len(overview))

# --------------------------------------------------
# 4. Long-term pressure overview
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

for col in ["TP2", "TP3", "H1", "Reservoirs"]:
    ax.plot(
        overview["timestamp"],
        overview[col],
        label=col,
        linewidth=0.8,
        alpha=0.8,
    )

ax.set_title("Industrial Compressor: Pressure Sensor Overview")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Sensor reading (recorded units)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "pressure_overview.png", dpi=150)
plt.close()

# --------------------------------------------------
# 5. Long-term temperature and current overview
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(
    overview["timestamp"],
    overview["Oil_temperature"],
    linewidth=0.8,
)
axes[0].set_title("Oil Temperature Over Time")
axes[0].set_ylabel("Oil temperature")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    overview["timestamp"],
    overview["Motor_current"],
    linewidth=0.8,
)
axes[1].set_title("Motor Current Over Time")
axes[1].set_xlabel("Timestamp")
axes[1].set_ylabel("Motor current")
axes[1].grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "temperature_current_overview.png", dpi=150)
plt.close()

# --------------------------------------------------
# 6. Select a short period for detailed investigation
# --------------------------------------------------
# Select the first six hours of valid data.
# Students can change this duration later.
start_time = overview["timestamp"].min()
end_time = start_time + pd.Timedelta(hours=6)

# Read the selected period from the original file.
# This avoids relying on the downsampled overview for detailed analysis.
detail_parts = []

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=required_columns,
    chunksize=chunk_size,
):
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"], errors="coerce"
    )

    chunk = chunk[
        (chunk["timestamp"] >= start_time)
        & (chunk["timestamp"] <= end_time)
    ]

    if chunk.empty:
        continue

    for col in sensor_columns:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        )

    detail_parts.append(chunk)

if detail_parts:
    detail = pd.concat(detail_parts, ignore_index=True)
    detail = detail.sort_values("timestamp")

    # Plot pressure separately from temperature and current.
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for col in ["TP2", "TP3", "Reservoirs"]:
        axes[0].plot(
            detail["timestamp"], detail[col],
            label=col, linewidth=0.9
        )

    axes[0].set_title("Detailed Pressure Behaviour")
    axes[0].set_ylabel("Recorded pressure units")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        detail["timestamp"], detail["Oil_temperature"],
        linewidth=0.9
    )
    axes[1].set_title("Detailed Oil Temperature")
    axes[1].set_ylabel("Recorded temperature units")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        detail["timestamp"], detail["Motor_current"],
        linewidth=0.9
    )
    axes[2].set_title("Detailed Motor Current")
    axes[2].set_ylabel("Recorded current units")
    axes[2].set_xlabel("Timestamp")
    axes[2].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "six_hour_sensor_detail.png", dpi=150)
    plt.close()

    print("\nDetailed period start:", start_time)
    print("Detailed period end:", end_time)
    print("Rows in detailed period:", len(detail))
else:
    print("No valid records found in the selected period.")

# --------------------------------------------------
# 7. Explore motor-current and pressure relationship
# --------------------------------------------------
relationship_data = overview[
    ["Motor_current", "TP2"]
].dropna()

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    relationship_data["Motor_current"],
    relationship_data["TP2"],
    alpha=0.25,
    s=8,
)

ax.set_title("Motor Current vs TP2 Pressure")
ax.set_xlabel("Motor current")
ax.set_ylabel("TP2 pressure")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "current_vs_tp2.png", dpi=150)
plt.close()

print("\nSaved plots in:", FIGURE_DIR)
print("Sensor visualization completed.")
