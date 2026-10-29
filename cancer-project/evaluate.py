
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the dataset
data = load_breast_cancer()

# Create the same reproducible test split
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)

# Load the Production model from MLflow Registry
model = mlflow.sklearn.load_model(
    "models:/BreastCancerClassifier/Production"
)

# Evaluate the model
with mlflow.start_run(run_name="production-model-evaluation"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log evaluation parameters and metrics
    mlflow.log_param("model_stage", "Production")
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"Production Model Accuracy: {accuracy:.4f}")
    print(f"Production Model ROC-AUC: {roc_auc:.4f}")
