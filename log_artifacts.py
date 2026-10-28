import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

mlflow.set_experiment("artifact-logging")

with mlflow.start_run(run_name="artifact-demo"):

    # 1. Create JSON evaluation report
    report = {
        "accuracy": 0.92,
        "f1": 0.89,
        "roc_auc": 0.95
    }

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    mlflow.log_artifact(
        "eval_report.json",
        artifact_path="reports"
    )

    # 2. Create training curve plot
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, 21)

    train_loss = (
        1 - 0.05 * epochs
        + np.random.normal(0, 0.01, 20)
    )

    val_loss = (
        1 - 0.04 * epochs
        + np.random.normal(0, 0.015, 20)
    )

    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training curve")

    fig.savefig(
        "training_curve.png",
        dpi=100,
        bbox_inches="tight"
    )

    mlflow.log_artifact(
        "training_curve.png",
        artifact_path="plots"
    )

    plt.close(fig)

    # 3. Create and log CSV artifact
    df = pd.DataFrame({
        "student": ["Ali", "Sara", "Ahmed"],
        "marks": [85, 92, 78]
    })

    df.to_csv("data_sample.csv", index=False)

    mlflow.log_artifact(
        "data_sample.csv",
        artifact_path="data"
    )

    # 4. Log evaluation metrics
    mlflow.log_metrics(report)

    print("Artifacts logged!")

