
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get the experiment
exp = client.get_experiment_by_name("sklearn-classifier")

if exp is None:
    print("Experiment not found. Check the experiment name.")
    raise SystemExit

print(f"Experiment ID: {exp.experiment_id}")

# Search and sort runs
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.accuracy > 0.92",
    order_by=["metrics.roc_auc ASC"],
    max_results=10
)

print(f"\nFound {len(runs)} qualifying runs:\n")

# Create a summary table
rows = []

for run in runs:
    rows.append({
        "run_id": run.info.run_id[:8],
        "status": run.info.status,
        "n_estimators": run.data.params.get("n_estimators"),
        "max_depth": run.data.params.get("max_depth"),
        "accuracy": round(run.data.metrics.get("accuracy", 0), 4),
        "roc_auc": round(run.data.metrics.get("roc_auc", 0), 4)
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False) if rows else "No qualifying runs found.")

if runs:
    best = runs[0]

    print(f"\nBest run_id: {best.info.run_id}")
    print(f"Best params: {best.data.params}")

    # Bonus: Generate a Markdown model card
    print("\n# Best Model Card")
    print(f"- **Run ID:** {best.info.run_id}")
    print(f"- **Status:** {best.info.status}")
    print("\n## Parameters")
    for key, value in best.data.params.items():
        print(f"- **{key}:** {value}")

    print("\n## Metrics")
    for key, value in best.data.metrics.items():
        print(f"- **{key}:** {value:.4f}")

    # Try downloading the plots folder
    try:
        path = client.download_artifacts(
            best.info.run_id, "plots", "/tmp/artifacts"
        )
        print(f"\nArtifacts downloaded to: {path}")
    except Exception as e:
        print(f"\nPlots folder not found for this run: {e}")
