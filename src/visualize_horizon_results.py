from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
COMPARISON_PATH = PROJECT_DIR / "reports" / "rnn_horizon_comparison.csv"
LOSS_HISTORY_PATH = PROJECT_DIR / "reports" / "rnn_horizon_loss_history.csv"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
ERROR_FIGURE_PATH = FIGURE_DIR / "rnn_horizon_errors.png"
LOSS_FIGURE_PATH = FIGURE_DIR / "rnn_horizon_loss_curves.png"


def main() -> None:
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Horizon comparison not found: {COMPARISON_PATH}\n"
            "Run `python .\\src\\train_rnn_horizons.py` first."
        )
    if not LOSS_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Horizon loss history not found: {LOSS_HISTORY_PATH}\n"
            "Run `python .\\src\\train_rnn_horizons.py` first."
        )

    comparison = pd.read_csv(COMPARISON_PATH)
    history = pd.read_csv(LOSS_HISTORY_PATH)
    required_metrics = {
        "horizon",
        "model_mae",
        "model_rmse",
        "persistence_mae",
        "persistence_rmse",
    }
    missing_metrics = required_metrics.difference(comparison.columns)
    if missing_metrics:
        raise ValueError(
            "Horizon comparison is missing columns: "
            + ", ".join(sorted(missing_metrics))
        )
    required_history = {"horizon", "epoch", "training_loss", "validation_loss"}
    missing_history = required_history.difference(history.columns)
    if missing_history:
        raise ValueError(
            "Horizon loss history is missing columns: "
            + ", ".join(sorted(missing_history))
        )
    if comparison.empty or history.empty:
        raise ValueError("Horizon metrics and loss history must not be empty.")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comparison.plot(
        x="horizon",
        y=["model_mae", "persistence_mae"],
        kind="bar",
        ax=axes[0],
        color=["tab:blue", "tab:orange"],
    )
    axes[0].set_title("Test MAE by Forecast Horizon")
    axes[0].set_xlabel("Forecast horizon")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    comparison.plot(
        x="horizon",
        y=["model_rmse", "persistence_rmse"],
        kind="bar",
        ax=axes[1],
        color=["tab:blue", "tab:orange"],
    )
    axes[1].set_title("Test RMSE by Forecast Horizon")
    axes[1].set_xlabel("Forecast horizon")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ERROR_FIGURE_PATH, dpi=150)
    plt.close(fig)

    horizons = history["horizon"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(9, max(4, 3 * len(horizons))),
        squeeze=False,
    )
    for ax, horizon in zip(axes[:, 0], horizons):
        horizon_history = history[history["horizon"] == horizon]
        ax.plot(
            horizon_history["epoch"],
            horizon_history["training_loss"],
            label="Training loss",
        )
        ax.plot(
            horizon_history["epoch"],
            horizon_history["validation_loss"],
            label="Validation loss",
        )
        ax.set_title(f"{horizon} Forecast Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_FIGURE_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved horizon error charts to: {ERROR_FIGURE_PATH}")
    print(f"Saved horizon loss curves to: {LOSS_FIGURE_PATH}")


if __name__ == "__main__":
    main()
