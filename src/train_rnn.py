from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "feature_engineered.csv"
MODEL_PATH = PROJECT_DIR / "models" / "oil_temperature_rnn.pt"
LOSS_HISTORY_PATH = PROJECT_DIR / "reports" / "rnn_loss_history.csv"
COMPARISON_PATH = PROJECT_DIR / "reports" / "rnn_tuning_comparison.csv"

TARGET_COLUMN = "Oil_temperature"
SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 64
EXPERIMENTS = [
    {"name": "baseline", "sequence_length": 12, "hidden_size": 64, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "longer_sequence", "sequence_length": 24, "hidden_size": 64, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "smaller_units", "sequence_length": 12, "hidden_size": 32, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "higher_dropout", "sequence_length": 12, "hidden_size": 64, "dropout": 0.2, "learning_rate": 1e-3},
    {"name": "lower_learning_rate", "sequence_length": 12, "hidden_size": 64, "dropout": 0.1, "learning_rate": 5e-4},
]
MAX_TRAINING_WINDOWS = 100_000
BATCH_SIZE = 512
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42


class VanillaRNNForecaster(nn.Module):
    """A one-layer tanh RNN that predicts from the last hidden output."""

    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_outputs, _ = self.rnn(sequence)
        last_output = sequence_outputs[:, -1, :]
        return self.output_layer(self.dropout(last_output)).squeeze(-1)


def make_sequence_batch(
    rows: np.ndarray,
    target_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Return preceding rows for each target, with train-fitted scaling."""
    offsets = np.arange(sequence_length)
    batch = rows[target_indices[:, None] - sequence_length + offsets]
    batch = np.where(np.isfinite(batch), batch, feature_means[None, None, :])
    batch = (batch - feature_means[None, None, :]) / feature_scales[None, None, :]
    return batch.astype(np.float32)


def predict_in_batches(
    model: VanillaRNNForecaster,
    rows: np.ndarray,
    target_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: torch.device,
    sequence_length: int,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(target_indices), BATCH_SIZE):
            batch_indices = target_indices[start : start + BATCH_SIZE]
            batch = make_sequence_batch(
                rows, batch_indices, feature_means, feature_scales, sequence_length
            )
            inputs = torch.from_numpy(batch).to(device)
            normalized_predictions = model(inputs).cpu().numpy()
            predictions.append(normalized_predictions * target_scale + target_mean)
    return np.concatenate(predictions)


def print_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> None:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name} MAE:  {mean_absolute_error(actual, predicted):.4f}")
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} R2:   {r2_score(actual, predicted):.4f}")


def train_model(
    rows: np.ndarray,
    target: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: torch.device,
    config: dict[str, float | int | str],
) -> tuple[VanillaRNNForecaster, dict[str, list[float]]]:
    """Train with chronological validation and restore the best epoch."""
    torch.manual_seed(RANDOM_SEED)
    sequence_length = int(config["sequence_length"])
    model = VanillaRNNForecaster(
        rows.shape[1],
        int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    loss_function = nn.MSELoss()
    history = {"training_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    normalized_target = (target - target_mean) / target_scale

    for epoch in range(MAX_EPOCHS):
        model.train()
        shuffled_indices = np.random.default_rng(
            RANDOM_SEED + epoch
        ).permutation(training_indices)
        total_training_loss = 0.0

        for start in range(0, len(shuffled_indices), BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            batch = make_sequence_batch(
                rows,
                batch_indices,
                feature_means,
                feature_scales,
                sequence_length,
            )
            inputs = torch.from_numpy(batch).to(device)
            labels = torch.from_numpy(
                normalized_target[batch_indices].astype(np.float32)
            ).to(device)

            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item() * len(batch_indices)

        model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_indices), BATCH_SIZE):
                batch_indices = validation_indices[start : start + BATCH_SIZE]
                batch = make_sequence_batch(
                    rows,
                    batch_indices,
                    feature_means,
                    feature_scales,
                    sequence_length,
                )
                inputs = torch.from_numpy(batch).to(device)
                labels = torch.from_numpy(
                    normalized_target[batch_indices].astype(np.float32)
                ).to(device)
                loss = loss_function(model(inputs), labels)
                total_validation_loss += loss.item() * len(batch_indices)

        training_loss = total_training_loss / len(training_indices)
        validation_loss = total_validation_loss / len(validation_indices)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} "
            f"- training loss: {training_loss:.6f} "
            f"- validation loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation loss did not improve.")
                break

    if best_model_state is None:
        raise RuntimeError("RNN training did not produce a valid model.")

    model.load_state_dict(best_model_state)
    return model, history


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

    feature_data = (
        data.drop(columns=["timestamp"])
        .select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    if TARGET_COLUMN not in feature_data.columns:
        raise ValueError(f"Target column must be numeric: {TARGET_COLUMN}")

    rows = feature_data.to_numpy(dtype=np.float32)
    target = feature_data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    feature_names = feature_data.columns.tolist()
    del feature_data, data

    train_end = int(len(rows) * 0.70)
    validation_end = int(len(rows) * 0.85)
    max_sequence_length = max(int(config["sequence_length"]) for config in EXPERIMENTS)
    if train_end <= max_sequence_length or validation_end <= train_end:
        raise ValueError("Not enough rows for the configured splits and sequence.")

    training_rows = rows[:train_end]
    finite_training_values = np.isfinite(training_rows)
    finite_counts = finite_training_values.sum(axis=0)
    feature_means = np.zeros(rows.shape[1], dtype=np.float32)
    has_training_values = finite_counts > 0
    feature_means[has_training_values] = np.nanmean(
        np.where(finite_training_values, training_rows, np.nan),
        axis=0,
    )[has_training_values]
    feature_scales = np.ones(rows.shape[1], dtype=np.float32)
    feature_scales[has_training_values] = np.nanstd(
        np.where(finite_training_values, training_rows, np.nan),
        axis=0,
    )[has_training_values]
    feature_scales[
        ~np.isfinite(feature_scales) | (feature_scales < 1e-6)
    ] = 1.0
    del training_rows, finite_training_values

    training_targets = target[:train_end]
    training_targets = training_targets[np.isfinite(training_targets)]
    if not len(training_targets):
        raise ValueError("No valid training targets were found.")
    target_mean = float(training_targets.mean())
    target_scale = float(training_targets.std())
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0

    training_indices = np.arange(max_sequence_length, train_end)
    validation_indices = np.arange(train_end, validation_end)
    test_indices = np.arange(validation_end, len(rows))
    training_indices = training_indices[np.isfinite(target[training_indices])]
    validation_indices = validation_indices[
        np.isfinite(target[validation_indices])
    ]
    test_indices = test_indices[np.isfinite(target[test_indices])]
    test_indices = test_indices[np.isfinite(target[test_indices - 1])]

    if not len(training_indices):
        raise ValueError("No valid training targets remain after windowing.")
    if not len(validation_indices) or not len(test_indices):
        raise ValueError("Validation and test splits must both have valid targets.")

    if len(training_indices) > MAX_TRAINING_WINDOWS:
        sample_positions = np.linspace(
            0, len(training_indices) - 1, num=MAX_TRAINING_WINDOWS, dtype=int
        )
        training_indices = training_indices[sample_positions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training windows:   {len(training_indices):,}")
    print(f"Validation windows: {len(validation_indices):,}")
    print(f"Test windows:       {len(test_indices):,}")
    print(f"Input features:     {len(feature_names)}")
    print(f"Maximum sequence length: {max_sequence_length}")
    print(f"Experiments:              {len(EXPERIMENTS)}")

    experiment_results = []
    best_model = None
    best_history = None
    best_config = None
    best_validation_rmse = float("inf")
    for experiment in EXPERIMENTS:
        print(f"\nRunning experiment: {experiment['name']}")
        model, history = train_model(
            rows,
            target,
            training_indices,
            validation_indices,
            feature_means,
            feature_scales,
            target_mean,
            target_scale,
            device,
            experiment,
        )
        validation_predictions = predict_in_batches(
            model,
            rows,
            validation_indices,
            feature_means,
            feature_scales,
            target_mean,
            target_scale,
            device,
            int(experiment["sequence_length"]),
        )
        validation_actual = target[validation_indices]
        validation_mae = mean_absolute_error(
            validation_actual, validation_predictions
        )
        validation_rmse = np.sqrt(
            mean_squared_error(validation_actual, validation_predictions)
        )
        validation_r2 = r2_score(validation_actual, validation_predictions)
        experiment_results.append(
            {
                **experiment,
                "best_validation_loss": min(history["validation_loss"]),
                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
                "validation_r2": validation_r2,
            }
        )
        if validation_rmse < best_validation_rmse:
            best_model = deepcopy(model)
            best_history = history
            best_config = experiment.copy()
            best_validation_rmse = validation_rmse

    comparison = pd.DataFrame(experiment_results).sort_values(
        "validation_rmse"
    )
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    print("\nExperiment comparison (ranked by validation RMSE):")
    print(comparison.to_string(index=False))
    print(f"Saved comparison table to: {COMPARISON_PATH}")

    if best_model is None or best_history is None or best_config is None:
        raise RuntimeError("No RNN experiment produced a selected model.")

    test_predictions = predict_in_batches(
        best_model,
        rows,
        test_indices,
        feature_means,
        feature_scales,
        target_mean,
        target_scale,
        device,
        int(best_config["sequence_length"]),
    )
    test_actual = target[test_indices]
    print("\nTest results:")
    print(f"Selected configuration: {best_config}")
    print_metrics("Tuned vanilla RNN", test_actual, test_predictions)
    print_metrics("Persistence baseline", test_actual, target[test_indices - 1])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "feature_names": feature_names,
            "feature_means": torch.from_numpy(feature_means),
            "feature_scales": torch.from_numpy(feature_scales),
            "target_mean": target_mean,
            "target_scale": target_scale,
            **best_config,
        },
        MODEL_PATH,
    )

    LOSS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(best_history["training_loss"]) + 1),
            "training_loss": best_history["training_loss"],
            "validation_loss": best_history["validation_loss"],
        }
    ).to_csv(LOSS_HISTORY_PATH, index=False)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved loss history to: {LOSS_HISTORY_PATH}")
    print(f"Saved experiment comparison to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
