
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Connect to MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Load the Production model from Registry
model_uri = "models:/BreastCancerClassifier/3"
model = mlflow.pyfunc.load_model(model_uri)

# Load sample data
data = load_breast_cancer(as_frame=True)
X = data.data

# Select 5 samples for prediction
sample_data = X.head(5).to_numpy()

# Make predictions
predictions = model.predict(sample_data)

print("Sample input:")
print(sample_data)

print("\nPredictions:")
print(predictions)
