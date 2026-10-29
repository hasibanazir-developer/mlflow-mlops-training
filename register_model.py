import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5001")

client = MlflowClient()

BEST_RUN_ID = "bd9c95d0984446e8b5b1b62aed9f42f9"
MODEL_NAME = "BreastCancerClassifier"

model_uri = f"runs:/{BEST_RUN_ID}/model"

result = mlflow.register_model(model_uri, MODEL_NAME)

print(f"Registered: {MODEL_NAME} v{result.version}")
print(f"Current stage: {result.current_stage}")

client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description=(
        "Random Forest on Breast Cancer dataset. "
        "Accuracy > 0.92, ROC-AUC > 0.97."
    )
)

client.set_registered_model_tag(
    MODEL_NAME, "framework", "sklearn"
)
client.set_registered_model_tag(
    MODEL_NAME, "owner", "mlops-lab-student"
)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Staging",
    archive_existing_versions=False
)

print("Moved to Staging!")
