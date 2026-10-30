import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_OUTPUTS = {
    "mlp": {
        "history": PROJECT_DIR / "reports" / "mlp_loss_history.csv",
        "figure": PROJECT_DIR / "reports" / "figures" / "mlp_loss_curve.png",
    },
    "rnn": {
        "history": PROJECT_DIR / "reports" / "rnn_loss_history.csv",
        "figure": PROJECT_DIR / "reports" / "figures" / "rnn_loss_curve.png",
    },
    "lstm": {
        "history": PROJECT_DIR / "reports" / "lstm_loss_history.csv",
        "figure": PROJECT_DIR / "reports" / "figures" / "lstm_loss_curve.png",
    },
    "gru": {
        "history": PROJECT_DIR / "reports" / "gru_loss_history.csv",
        "figure": PROJECT_DIR / "reports" / "figures" / "gru_loss_curve.png",
    },
    "cnn": {
        "history": PROJECT_DIR / "reports" / "cnn_loss_history.csv",
        "figure": PROJECT_DIR / "reports" / "figures" / "cnn_loss_curve.png",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training and validation losses for a forecasting model."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_OUTPUTS,
        default="mlp",
        help="Model loss history to plot (default: mlp).",
    )
    model_name = parser.parse_args().model
    model_outputs = MODEL_OUTPUTS[model_name]
    loss_history_path = model_outputs["history"]
    figure_path = model_outputs["figure"]

    if not loss_history_path.exists():
        raise FileNotFoundError(
            f"Loss history not found: {loss_history_path}\n"
            "Run the corresponding model training script first."
        )

    history = pd.read_csv(loss_history_path)
    required_columns = {"epoch", "training_loss", "validation_loss"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(
            "Loss history is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if history.empty:
        raise ValueError(f"Loss history is empty: {loss_history_path}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["training_loss"], label="Training loss")
    ax.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title(f"{model_name.upper()} Forecasting Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to: {figure_path}")


if __name__ == "__main__":
    main()
