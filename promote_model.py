
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5001")

client = MlflowClient()

EXPERIMENT_NAME = "sklearn-classifier"
MODEL_NAME = "BreastCancerClassifier"

# Find the experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Search runs where ROC-AUC is greater than 0.97
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.roc_auc > 0.97",
    order_by=["metrics.roc_auc DESC"],
    max_results=100
)

if not runs:
    print("No qualifying runs found.")
    raise SystemExit

# Select the run with the highest ROC-AUC
best_run = runs[0]
print("Best run:", best_run.info.run_id)
print("ROC-AUC:", best_run.data.metrics["roc_auc"])

# Check whether a Production version already exists
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

production_exists = any(
    version.current_stage == "Production"
    for version in versions
)

# Register the selected model
model_uri = f"runs:/{best_run.info.run_id}/model"
registered = mlflow.register_model(model_uri, MODEL_NAME)

print("Registered version:", registered.version)

# Promote only if no Production version exists
if not production_exists:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    print("Model promoted to Production.")
else:
    print("Production model already exists. No promotion performed.")
