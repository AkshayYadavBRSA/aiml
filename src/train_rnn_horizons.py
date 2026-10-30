from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "feature_engineered.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
COMPARISON_PATH = REPORT_DIR / "rnn_horizon_comparison.csv"
LOSS_HISTORY_PATH = REPORT_DIR / "rnn_horizon_loss_history.csv"

TARGET_COLUMN = "Oil_temperature"
HORIZONS = {
    "short_10s": pd.Timedelta(seconds=10),
    "medium_1m": pd.Timedelta(minutes=1),
    "long_5m": pd.Timedelta(minutes=5),
}
SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 64
BATCH_SIZE = 512
MAX_TRAINING_WINDOWS = 100_000
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42


class HorizonRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(sequence)
        return self.output_layer(outputs[:, -1, :]).squeeze(-1)


def build_horizon_targets(
    timestamps: np.ndarray,
    origin_indices: np.ndarray,
    horizon: pd.Timedelta,
) -> np.ndarray:
    """Map each observation time to the first row at or after its horizon."""
    target_times = timestamps[origin_indices] + horizon.to_timedelta64()
    return np.searchsorted(timestamps, target_times, side="left")


def make_batch(
    rows: np.ndarray,
    origin_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
) -> np.ndarray:
    offsets = np.arange(SEQUENCE_LENGTH)
    batch = rows[
        origin_indices[:, None] - SEQUENCE_LENGTH + 1 + offsets
    ]
    batch = np.where(np.isfinite(batch), batch, feature_means[None, None, :])
    batch = (batch - feature_means[None, None, :]) / feature_scales[None, None, :]
    return batch.astype(np.float32)


def predict(
    model: HorizonRNN,
    rows: np.ndarray,
    origin_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(origin_indices), BATCH_SIZE):
            batch_origins = origin_indices[start : start + BATCH_SIZE]
            inputs = torch.from_numpy(
                make_batch(rows, batch_origins, feature_means, feature_scales)
            ).to(device)
            normalized_predictions = model(inputs).cpu().numpy()
            predictions.append(normalized_predictions * target_scale + target_mean)
    return np.concatenate(predictions)


def print_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> None:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"  {name} MAE:  {mean_absolute_error(actual, predicted):.4f}")
    print(f"  {name} RMSE: {rmse:.4f}")
    print(f"  {name} R2:   {r2_score(actual, predicted):.4f}")


def train_horizon_model(
    rows: np.ndarray,
    target: np.ndarray,
    training_origins: np.ndarray,
    training_target_indices: np.ndarray,
    validation_origins: np.ndarray,
    validation_target_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    device: torch.device,
    horizon_name: str,
) -> tuple[HorizonRNN, dict[str, list[float]], float, float]:
    training_labels = target[training_target_indices]
    target_mean = float(training_labels.mean())
    target_scale = float(training_labels.std())
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0
    normalized_target = (target - target_mean) / target_scale

    torch.manual_seed(RANDOM_SEED)
    model = HorizonRNN(rows.shape[1], HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    history = {"training_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        shuffled_positions = np.random.default_rng(
            RANDOM_SEED + epoch
        ).permutation(len(training_origins))
        total_training_loss = 0.0

        for start in range(0, len(shuffled_positions), BATCH_SIZE):
            positions = shuffled_positions[start : start + BATCH_SIZE]
            batch_origins = training_origins[positions]
            batch_targets = training_target_indices[positions]
            inputs = torch.from_numpy(
                make_batch(rows, batch_origins, feature_means, feature_scales)
            ).to(device)
            labels = torch.from_numpy(
                normalized_target[batch_targets].astype(np.float32)
            ).to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item() * len(positions)

        model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_origins), BATCH_SIZE):
                batch_origins = validation_origins[start : start + BATCH_SIZE]
                batch_targets = validation_target_indices[
                    start : start + BATCH_SIZE
                ]
                inputs = torch.from_numpy(
                    make_batch(rows, batch_origins, feature_means, feature_scales)
                ).to(device)
                labels = torch.from_numpy(
                    normalized_target[batch_targets].astype(np.float32)
                ).to(device)
                loss = loss_function(model(inputs), labels)
                total_validation_loss += loss.item() * len(batch_origins)

        training_loss = total_training_loss / len(training_origins)
        validation_loss = total_validation_loss / len(validation_origins)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"  {horizon_name} epoch {epoch + 1:02d}/{MAX_EPOCHS} "
            f"- training loss: {training_loss:.6f} "
            f"- validation loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  {horizon_name}: early stopping.")
                break

    if best_state is None:
        raise RuntimeError(f"No valid RNN checkpoint for horizon {horizon_name}.")
    model.load_state_dict(best_state)
    return model, history, target_mean, target_scale


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed feature data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = (
        data.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    timestamps = data["timestamp"].to_numpy(dtype="datetime64[ns]")
    feature_data = (
        data.drop(columns=["timestamp"])
        .select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    if TARGET_COLUMN not in feature_data.columns:
        raise ValueError(f"Target column must be numeric: {TARGET_COLUMN}")
    feature_names = feature_data.columns.tolist()
    rows = feature_data.to_numpy(dtype=np.float32)
    target = feature_data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    del feature_data, data

    train_end = int(len(rows) * 0.70)
    validation_end = int(len(rows) * 0.85)
    if train_end <= SEQUENCE_LENGTH or validation_end <= train_end:
        raise ValueError("Not enough rows for chronological splits and sequences.")

    training_rows = rows[:train_end]
    finite_training_rows = np.isfinite(training_rows)
    feature_means = np.zeros(rows.shape[1], dtype=np.float32)
    has_values = finite_training_rows.any(axis=0)
    feature_means[has_values] = np.nanmean(
        np.where(finite_training_rows, training_rows, np.nan), axis=0
    )[has_values]
    feature_scales = np.ones(rows.shape[1], dtype=np.float32)
    feature_scales[has_values] = np.nanstd(
        np.where(finite_training_rows, training_rows, np.nan), axis=0
    )[has_values]
    feature_scales[
        ~np.isfinite(feature_scales) | (feature_scales < 1e-6)
    ] = 1.0
    del training_rows, finite_training_rows

    origin_indices = np.arange(SEQUENCE_LENGTH - 1, len(rows))
    result_rows = []
    history_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Forecasting type: direct (a separately trained model per horizon)")
    print(f"Input sequence length: {SEQUENCE_LENGTH} rows")

    for horizon_name, horizon in HORIZONS.items():
        target_indices = build_horizon_targets(
            timestamps, origin_indices, horizon
        )
        valid = target_indices < len(rows)
        valid &= np.isfinite(target[np.minimum(target_indices, len(rows) - 1)])
        valid &= np.isfinite(target[origin_indices])
        origins = origin_indices[valid]
        future_indices = target_indices[valid]

        train_mask = future_indices < train_end
        validation_mask = (
            (future_indices >= train_end) & (future_indices < validation_end)
        )
        test_mask = future_indices >= validation_end
        train_origins = origins[train_mask]
        train_targets = future_indices[train_mask]
        validation_origins = origins[validation_mask]
        validation_targets = future_indices[validation_mask]
        test_origins = origins[test_mask]
        test_targets = future_indices[test_mask]

        if not len(train_origins) or not len(validation_origins) or not len(test_origins):
            raise ValueError(f"Insufficient samples for horizon {horizon_name}.")

        if len(train_origins) > MAX_TRAINING_WINDOWS:
            sample_positions = np.linspace(
                0, len(train_origins) - 1, MAX_TRAINING_WINDOWS, dtype=int
            )
            train_origins = train_origins[sample_positions]
            train_targets = train_targets[sample_positions]

        actual_leads = (
            timestamps[future_indices] - timestamps[origins]
        ) / np.timedelta64(1, "s")
        print(
            f"\n{horizon_name}: requested {horizon}, "
            f"median observed lead {np.median(actual_leads):.1f}s"
        )
        model, history, target_mean, target_scale = train_horizon_model(
            rows,
            target,
            train_origins,
            train_targets,
            validation_origins,
            validation_targets,
            feature_means,
            feature_scales,
            device,
            horizon_name,
        )

        validation_predictions = predict(
            model,
            rows,
            validation_origins,
            feature_means,
            feature_scales,
            target_mean,
            target_scale,
            device,
        )
        validation_actual = target[validation_targets]
        print(f"  Validation {horizon_name} results:")
        print_metrics(horizon_name, validation_actual, validation_predictions)

        test_predictions = predict(
            model,
            rows,
            test_origins,
            feature_means,
            feature_scales,
            target_mean,
            target_scale,
            device,
        )
        test_actual = target[test_targets]
        persistence_predictions = target[test_origins]
        test_leads = (
            timestamps[test_targets] - timestamps[test_origins]
        ) / np.timedelta64(1, "s")
        result_rows.append(
            {
                "horizon": horizon_name,
                "requested_seconds": horizon.total_seconds(),
                "median_actual_seconds": float(np.median(test_leads)),
                "test_samples": len(test_targets),
                "model_mae": mean_absolute_error(test_actual, test_predictions),
                "model_rmse": np.sqrt(
                    mean_squared_error(test_actual, test_predictions)
                ),
                "model_r2": r2_score(test_actual, test_predictions),
                "persistence_mae": mean_absolute_error(
                    test_actual, persistence_predictions
                ),
                "persistence_rmse": np.sqrt(
                    mean_squared_error(test_actual, persistence_predictions)
                ),
                "persistence_r2": r2_score(
                    test_actual, persistence_predictions
                ),
            }
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"oil_temperature_rnn_{horizon_name}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_names": feature_names,
                "feature_means": torch.from_numpy(feature_means),
                "feature_scales": torch.from_numpy(feature_scales),
                "target_mean": target_mean,
                "target_scale": target_scale,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "horizon": horizon_name,
            },
            model_path,
        )
        for epoch, (training_loss, validation_loss) in enumerate(
            zip(history["training_loss"], history["validation_loss"]),
            start=1,
        ):
            history_rows.append(
                {
                    "horizon": horizon_name,
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                }
            )
        print(f"  Saved checkpoint: {model_path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)
    pd.DataFrame(history_rows).to_csv(LOSS_HISTORY_PATH, index=False)
    print("\nHeld-out test comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved horizon metrics to: {COMPARISON_PATH}")
    print(f"Saved loss history to: {LOSS_HISTORY_PATH}")


if __name__ == "__main__":
    main()
