import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Connect to the MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5001")

MODEL_NAME = "BreastCancerClassifier"
MODEL_STAGE = "Production"

# Load the registered Production model
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Option A: Framework-independent pyfunc model
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# Prepare 5 sample rows
data = load_breast_cancer()
sample = data.data[:5]

predictions = pyfunc_model.predict(sample)
print("pyfunc predictions:", predictions)

# Option B: sklearn model with predict_proba
sk_model = mlflow.sklearn.load_model(model_uri)
proba = sk_model.predict_proba(sample)

# Dataset class 0 = malignant, class 1 = benign
print("Probability of malignant:", proba[:, 0].round(3))
print("Probability of benign:", proba[:, 1].round(3))
