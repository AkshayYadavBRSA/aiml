from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "feature_engineered.csv"
MODEL_DIR = PROJECT_DIR / "models"
LOSS_HISTORY_PATH = PROJECT_DIR / "reports" / "mlp_loss_history.csv"
COMPARISON_PATH = PROJECT_DIR / "reports" / "mlp_tuning_comparison.csv"
TARGET_COLUMN = "Oil_temperature"
MAX_TRAIN_ROWS = 100_000
WINDOW_SIZE = 12
MAX_MLP_WINDOW_SIZE = 24
MLP_EXPERIMENTS = [
    {"name": "baseline", "sequence_length": 12, "hidden_units": 128, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "longer_sequence", "sequence_length": 24, "hidden_units": 128, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "fewer_units", "sequence_length": 12, "hidden_units": 64, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "higher_dropout", "sequence_length": 12, "hidden_units": 128, "dropout": 0.2, "learning_rate": 1e-3},
    {"name": "lower_learning_rate", "sequence_length": 12, "hidden_units": 128, "dropout": 0.1, "learning_rate": 5e-4},
]
PREDICTION_BATCH_SIZE = 2_048
MLP_BATCH_SIZE = 2_048
MLP_MAX_EPOCHS = 30
MLP_PATIENCE = 5


class MLPForecastingModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_units: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


def make_sliding_windows(
    features: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Create windows whose final row precedes the row being predicted."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(features) <= window_size:
        raise ValueError(
            f"Need more than {window_size} rows to create sliding windows; "
            f"found {len(features)}."
        )

    return np.lib.stride_tricks.sliding_window_view(
        features, window_shape=window_size, axis=0
    ).transpose(0, 2, 1)


def flatten_windows(
    windows: np.ndarray,
    target_indices: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Flatten past-only windows for the current scikit-learn estimator."""
    window_indices = target_indices - MAX_MLP_WINDOW_SIZE
    selected_windows = windows[window_indices, -window_size:, :]
    return selected_windows.reshape(len(target_indices), -1)


def prepare_mlp_batch(
    windows: np.ndarray,
    target_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    batch = windows[target_indices - MAX_MLP_WINDOW_SIZE, -sequence_length:, :]
    batch = np.where(np.isfinite(batch), batch, feature_means)
    batch = (batch - feature_means) / feature_scales
    return batch.reshape(len(target_indices), -1).astype(np.float32)


def predict_in_batches(
    model: HistGradientBoostingRegressor,
    windows: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(target_indices), PREDICTION_BATCH_SIZE):
        batch_indices = target_indices[start : start + PREDICTION_BATCH_SIZE]
        X_batch = flatten_windows(windows, batch_indices, WINDOW_SIZE)
        predictions.append(model.predict(X_batch))
    return np.concatenate(predictions)


def train_mlp(
    windows: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    feature_names: list[str],
    config: dict[str, float | int | str],
) -> tuple[MLPForecastingModel, dict[str, list[float]]]:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = int(config["sequence_length"])
    hidden_units = int(config["hidden_units"])
    dropout = float(config["dropout"])
    model = MLPForecastingModel(
        sequence_length * len(feature_names), hidden_units, dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"])
    )
    loss_function = nn.MSELoss()
    history = {"train": [], "validation": []}
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MLP_MAX_EPOCHS):
        model.train()
        shuffled_indices = np.random.default_rng(42 + epoch).permutation(fit_indices)
        total_train_loss = 0.0
        for start in range(0, len(shuffled_indices), MLP_BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + MLP_BATCH_SIZE]
            batch_features = prepare_mlp_batch(
                windows,
                batch_indices,
                feature_means,
                feature_scales,
                sequence_length,
            )
            inputs = torch.from_numpy(batch_features).to(device)
            labels = torch.from_numpy(target[batch_indices]).to(device)

            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_indices)

        model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_indices), PREDICTION_BATCH_SIZE):
                batch_indices = validation_indices[
                    start : start + PREDICTION_BATCH_SIZE
                ]
                batch_features = prepare_mlp_batch(
                    windows,
                    batch_indices,
                    feature_means,
                    feature_scales,
                    sequence_length,
                )
                inputs = torch.from_numpy(batch_features).to(device)
                labels = torch.from_numpy(target[batch_indices]).to(device)
                loss = loss_function(model(inputs), labels)
                total_validation_loss += loss.item() * len(batch_indices)

        train_loss = total_train_loss / len(fit_indices)
        validation_loss = total_validation_loss / len(validation_indices)
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        print(
            f"MLP epoch {epoch + 1:02d}/{MLP_MAX_EPOCHS} "
            f"- train loss: {train_loss:.6f} "
            f"- validation loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= MLP_PATIENCE:
                print("MLP early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    return model, history


def mlp_predict_in_batches(
    model: MLPForecastingModel,
    windows: np.ndarray,
    target_indices: np.ndarray,
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(target_indices), PREDICTION_BATCH_SIZE):
            batch_indices = target_indices[
                start : start + PREDICTION_BATCH_SIZE
            ]
            batch_features = prepare_mlp_batch(
                windows,
                batch_indices,
                feature_means,
                feature_scales,
                sequence_length,
            )
            inputs = torch.from_numpy(batch_features).to(device)
            predictions.append(model(inputs).cpu().numpy())
    return np.concatenate(predictions)


def print_metrics(
    name: str,
    actual: pd.Series,
    predicted: np.ndarray,
) -> None:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name} MAE:  {mean_absolute_error(actual, predicted):.4f}")
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} R²:   {r2_score(actual, predicted):.4f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {DATA_PATH}\n"
            "Place the processed feature dataset at this path before training."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    # Assign rows chronologically; each label is the target at that row and
    # its input window contains only the preceding rows.
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    feature_df = (
        df.drop(columns=["timestamp"])
        .select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    if TARGET_COLUMN not in feature_df.columns:
        raise ValueError(f"Target column is not numeric: {TARGET_COLUMN}")

    feature_names = feature_df.columns.tolist()
    features = feature_df.to_numpy(dtype=np.float32)
    target = feature_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    training_features = features[:train_end]
    finite_counts = np.isfinite(training_features).sum(axis=0)
    feature_means = np.zeros(len(feature_names), dtype=np.float32)
    valid_features = finite_counts > 0
    feature_means[valid_features] = np.nanmean(
        np.where(np.isfinite(training_features), training_features, np.nan),
        axis=0,
    )[valid_features]
    feature_scales = np.ones(len(feature_names), dtype=np.float32)
    feature_scales[valid_features] = np.nanstd(
        np.where(np.isfinite(training_features), training_features, np.nan),
        axis=0,
    )[valid_features]
    feature_scales[
        ~np.isfinite(feature_scales) | (feature_scales < 1e-6)
    ] = 1.0
    del training_features
    del feature_df, df

    windows = make_sliding_windows(features, MAX_MLP_WINDOW_SIZE)
    del features

    train_indices = np.arange(MAX_MLP_WINDOW_SIZE, train_end)
    validation_indices = np.arange(train_end, validation_end)
    test_indices = np.arange(validation_end, len(target))

    train_indices = train_indices[np.isfinite(target[train_indices])]
    validation_indices = validation_indices[
        np.isfinite(target[validation_indices])
    ]
    test_indices = test_indices[np.isfinite(target[test_indices])]
    test_indices = test_indices[
        np.isfinite(target[test_indices - 1])
    ]

    if not len(train_indices):
        raise ValueError("No valid training targets remain after windowing.")
    if not len(validation_indices) or not len(test_indices):
        raise ValueError(
            "The dataset must contain valid targets in both validation and "
            "test splits."
        )

    # Spread the training sample across the entire training period to keep
    # fitting manageable while retaining its chronological coverage.
    if len(train_indices) > MAX_TRAIN_ROWS:
        sample_positions = np.linspace(
            0, len(train_indices) - 1, num=MAX_TRAIN_ROWS, dtype=int
        )
        fit_indices = train_indices[sample_positions]
    else:
        fit_indices = train_indices

    X_fit = flatten_windows(windows, fit_indices, WINDOW_SIZE)
    y_fit = target[fit_indices]

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_fit, y_fit)
    training_rows_used = len(X_fit)
    flattened_feature_count = X_fit.shape[1]
    del X_fit, y_fit

    print(f"Training rows used: {training_rows_used:,}")
    print(f"Validation rows:    {len(validation_indices):,}")
    print(f"Test rows:          {len(test_indices):,}")
    print(f"Source feature count: {len(feature_names)}")
    print(f"MLP window sizes:    {[c['sequence_length'] for c in MLP_EXPERIMENTS]}")
    print(f"Flattened feature count: {flattened_feature_count}")

    print("\nValidation results:")
    validation_predictions = predict_in_batches(
        model, windows, validation_indices
    )
    print_metrics(
        "Validation", target[validation_indices], validation_predictions
    )

    print("\nTest results:")
    test_predictions = predict_in_batches(model, windows, test_indices)
    test_actual = target[test_indices]
    print_metrics("Model", test_actual, test_predictions)

    persistence_predictions = target[test_indices - 1]
    print_metrics("Persistence baseline", test_actual, persistence_predictions)

    mlp_results = []
    best_mlp = None
    best_mlp_history = None
    best_mlp_config = None
    best_mlp_validation_rmse = float("inf")
    for experiment in MLP_EXPERIMENTS:
        print(f"\nRunning MLP experiment: {experiment['name']}")
        mlp_model, mlp_history = train_mlp(
            windows,
            target,
            fit_indices,
            validation_indices,
            feature_means,
            feature_scales,
            feature_names,
            experiment,
        )
        mlp_validation_predictions = mlp_predict_in_batches(
            mlp_model,
            windows,
            validation_indices,
            feature_means,
            feature_scales,
            int(experiment["sequence_length"]),
        )
        validation_actual = target[validation_indices]
        validation_mae = mean_absolute_error(
            validation_actual, mlp_validation_predictions
        )
        validation_rmse = np.sqrt(
            mean_squared_error(validation_actual, mlp_validation_predictions)
        )
        validation_r2 = r2_score(validation_actual, mlp_validation_predictions)
        mlp_results.append(
            {
                **experiment,
                "best_validation_loss": min(mlp_history["validation"]),
                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
                "validation_r2": validation_r2,
            }
        )
        if validation_rmse < best_mlp_validation_rmse:
            best_mlp = deepcopy(mlp_model)
            best_mlp_history = mlp_history
            best_mlp_config = experiment.copy()
            best_mlp_validation_rmse = validation_rmse

    mlp_comparison = pd.DataFrame(mlp_results).sort_values("validation_rmse")
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlp_comparison.to_csv(COMPARISON_PATH, index=False)
    print("\nMLP experiment comparison (ranked by validation RMSE):")
    print(mlp_comparison.to_string(index=False))
    print(f"Saved MLP comparison table to: {COMPARISON_PATH}")
    if best_mlp is None or best_mlp_history is None or best_mlp_config is None:
        raise RuntimeError("No MLP experiment produced a selected model.")

    print(f"\nSelected MLP configuration: {best_mlp_config}")
    mlp_test_predictions = mlp_predict_in_batches(
        best_mlp,
        windows,
        test_indices,
        feature_means,
        feature_scales,
        int(best_mlp_config["sequence_length"]),
    )
    print("\nSelected MLP test results:")
    print_metrics("Tuned MLP", test_actual, mlp_test_predictions)

    LOSS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(best_mlp_history["train"]) + 1),
            "training_loss": best_mlp_history["train"],
            "validation_loss": best_mlp_history["validation"],
        }
    ).to_csv(LOSS_HISTORY_PATH, index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlp_model_path = MODEL_DIR / "oil_temperature_mlp.pt"
    torch.save(
        {
            "model_state_dict": best_mlp.state_dict(),
            "feature_means": torch.from_numpy(feature_means),
            "feature_scales": torch.from_numpy(feature_scales),
            "feature_names": feature_names,
            **best_mlp_config,
        },
        mlp_model_path,
    )
    print(f"Saved tuned MLP model to: {mlp_model_path}")
    print(f"Saved selected MLP loss history to: {LOSS_HISTORY_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "oil_temperature_next_step.joblib"
    dump(model, model_path)
    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()