import mlflow
import mlflow.sklearn
import sklearn
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Create experiment
mlflow.set_experiment("sklearn-classifier")

# Hyperparameters
params = {
    "n_estimators": 50,
    "max_depth": 3,
    "min_samples_split": 2,
    "random_state": 42
}

with mlflow.start_run():
    # 1. Load and split data
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )

    # 2. Log dataset information
    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("dataset_version", sklearn.__version__)
    mlflow.log_param("dataset_shape", str(data.data.shape))
    mlflow.log_param("n_features", data.data.shape[1])
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))

    # 3. Log hyperparameters
    mlflow.log_params(params)

    # 4. Cross-validation on training data
    cv_scores = cross_val_score(
        RandomForestClassifier(**params),
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mlflow.log_metrics({
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std()
    })

    # 5. Train final model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 6. Evaluate on test data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    mlflow.log_metrics(metrics)

    # 7. Create feature importance plot
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [data.feature_names[i] for i in top_idx],
        importances[top_idx]
    )
    ax.set_title("Top 10 Feature Importances")
    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=100)

    mlflow.log_artifact(
        "feature_importance.png",
        artifact_path="plots"
    )
    plt.close(fig)

    # 8. Log serialized model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Test metrics:", {
        k: round(v, 4) for k, v in metrics.items()
    })
    print("CV mean accuracy:", round(cv_scores.mean(), 4))
    print("CV std accuracy:", round(cv_scores.std(), 4))
    print("Pipeline completed!")
