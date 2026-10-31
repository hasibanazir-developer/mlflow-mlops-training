import os
import hashlib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np


# ---------------- CONFIGURATION ----------------
EXPERIMENT = "mlops-pipeline"
MODEL_NAME = "ProductionClassifier"

ACCURACY_GATE = float(os.getenv("ACCURACY_GATE", "0.999"))

IS_CI = os.getenv("CI", "false").lower() == "true"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001"
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

# ---------------- LOAD AND VERSION DATA ----------------
data = load_breast_cancer()
X, y = data.data, data.target

# Create an MD5 hash to identify this exact dataset content
dataset_hash = hashlib.md5(
    X.tobytes() + y.tobytes()
).hexdigest()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)


# ---------------- MODEL CANDIDATES ----------------
candidates = [
    (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        False
    ),
    (
        "GradBoost",
        GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        False
    ),
    (
        "LogisticReg",
        LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        True
    ),
]

run_results = []

print(f"Dataset MD5: {dataset_hash}")
print("\nTraining candidate models...\n")


# ---------------- TRAIN AND LOG ----------------
for name, model, scaled in candidates:
    X_train = X_tr_s if scaled else X_tr
    X_test = X_te_s if scaled else X_te

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("scaled", scaled)
        mlflow.log_param("dataset_md5", dataset_hash)
        mlflow.set_tag("dataset_md5", dataset_hash)

        model.fit(X_train, y_tr)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_te, predictions)
        roc_auc = roc_auc_score(y_te, probabilities)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "roc_auc": roc_auc
        })

        # Use cloudpickle to avoid the skops trusted-type error
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        run_results.append({
            "name": name,
            "run_id": run.info.run_id,
            "accuracy": accuracy,
            "roc_auc": roc_auc
        })

        print(
            f"{name:15s} accuracy={accuracy:.4f} "
            f"ROC-AUC={roc_auc:.4f}"        )


# ---------------- AUTOMATIC MODEL SELECTION ----------------
best = max(run_results, key=lambda item: item["roc_auc"])

print(f"\nBest candidate: {best['name']}")
print(f"Best ROC-AUC: {best['roc_auc']:.4f}")
print(f"Accuracy gate: {ACCURACY_GATE:.4f}")

# ---------------- CI/CD PROMOTION GATE ----------------
if best["accuracy"] < ACCURACY_GATE:
    print(
        f"GATE FAILED: accuracy {best['accuracy']:.4f} "
        f"< {ACCURACY_GATE:.4f}. No promotion."
    )

elif IS_CI:
    print("\nCI MODE: Model evaluation completed.")
    print("Registry promotion skipped in GitHub Actions.")
    print("Training and evaluation: PASSED")

else:
    # Register the selected model
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_model_version_tag(
        MODEL_NAME, mv.version, "dataset_md5", dataset_hash
    )
    client.set_model_version_tag(
        MODEL_NAME, mv.version, "candidate", best["name"]
    )

    # Move model to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False
    )

    # Integration smoke tests
    staging_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{mv.version}"
    )

    test_X = X_te_s if best["name"] == "LogisticReg" else X_te

    test_predictions = staging_model.predict(test_X)
    test_probabilities = staging_model.predict_proba(test_X)

    tests_passed = (
        len(test_predictions) == len(y_te)
        and test_probabilities.shape == (len(y_te), 2)
        and np.isfinite(test_probabilities).all()
    )

    if tests_passed:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(
            f"\nPROMOTED {best['name']} v{mv.version} "
            "to Production."
        )
        print("Integration tests: PASSED")
    else:
        print("Integration tests FAILED. No Production promotion.")

print("\nPipeline finished.")
