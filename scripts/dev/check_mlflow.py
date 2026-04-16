import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("assetpulse-dev")

with mlflow.start_run(run_name="hello-mlflow"):
    mlflow.log_param("dataset", "FD001")
    mlflow.log_metric("rmse", 42.0)

print("Logged a test run to MLflow.")
