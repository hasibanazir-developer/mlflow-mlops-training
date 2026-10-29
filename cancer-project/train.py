
import argparse
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=6)
parser.add_argument("--test-size", type=float, default=0.2)
args = parser.parse_args()

with mlflow.start_run():
    data = load_breast_cancer()

    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data,
        data.target,
        test_size=args.test_size,
        random_state=42
    )

    mlflow.log_params(vars(args))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    mlflow.log_metrics({
        "accuracy": accuracy_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba)
    })

    mlflow.sklearn.log_model(
    model,
    name="model",
    serialization_format="cloudpickle"
)

    print("Accuracy:", accuracy_score(y_te, y_pred))
    print("ROC-AUC:", roc_auc_score(y_te, y_proba))
