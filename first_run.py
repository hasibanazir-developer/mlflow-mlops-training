import mlflow

mlflow.set_experiment("my-first-experiment")

with mlflow.start_run(run_name="baseline"):
    # Parameters: inputs
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Metrics: measured outputs
    for epoch in range(10):
        mlflow.log_metric("train_loss", 1.0 - epoch * 0.08, step=epoch)
        mlflow.log_metric("val_loss", 1.0 - epoch * 0.06, step=epoch)

    # Tags: extra information
    mlflow.set_tag("team", "mlops-lab")
    mlflow.set_tag("data_version", "v1.0")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
