
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Locate the dataset
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "MetroPT3(AirCompressor).csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# --------------------------------------------------
# 2. Read only the timestamp column
# --------------------------------------------------
print("Reading timestamps...")


df = pd.read_csv(
    DATA_PATH,
    usecols=["timestamp"]
)

print("Rows read:", len(df))

# --------------------------------------------------
# 3. Convert timestamps into datetime format
# --------------------------------------------------
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_count = df["timestamp"].isna().sum()

print("\nInvalid timestamps:", invalid_count)

# Exclude invalid timestamps for time-based analysis
valid_timestamps = df.loc[
    df["timestamp"].notna(), "timestamp"
]

# --------------------------------------------------
# 4. Check duplicate timestamps
# --------------------------------------------------
duplicate_count = valid_timestamps.duplicated().sum()

print("Duplicate timestamps:", duplicate_count)

# --------------------------------------------------
# 5. Check chronological ordering
# --------------------------------------------------
time_difference = valid_timestamps.diff()

out_of_order_count = (time_difference < pd.Timedelta(0)).sum()

print("Out-of-order transitions:", out_of_order_count)

# --------------------------------------------------
# 6. Sort timestamps for interval analysis
# --------------------------------------------------
sorted_timestamps = valid_timestamps.sort_values()

intervals = sorted_timestamps.diff().dropna()

# Remove zero or negative intervals from sampling analysis
positive_intervals = intervals[intervals > pd.Timedelta(0)]

print("\nEarliest timestamp:", sorted_timestamps.min())
print("Latest timestamp:", sorted_timestamps.max())

print("\nTime interval summary:")
print(positive_intervals.describe())

# --------------------------------------------------
# 7. Find the most common sampling intervals
# --------------------------------------------------
interval_seconds = positive_intervals.dt.total_seconds()

print("\nMost common positive intervals (seconds):")
print(
    interval_seconds
    .value_counts()
    .head(10)
)

# --------------------------------------------------
# 8. Inspect unusually large gaps
# --------------------------------------------------
if not positive_intervals.empty:
    median_interval = positive_intervals.median()
    

    large_gaps = positive_intervals[
    positive_intervals >
    max(
        pd.Timedelta(seconds=60),
        3 * median_interval
    )
]

    print("\nMedian positive interval:", median_interval)
    print("Large gaps over the inspection threshold:", len(large_gaps))
    print("Total duration of these gaps:", large_gaps.sum())
else:
    print("\nNot enough positive intervals to analyze gaps.")

print("\nTimestamp inspection completed.")
