
import hashlib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.atm_prediction.cleaning import clean_train
from src.atm_prediction.model import split_data, build_pipeline


# Load configuration
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# Calculate SHA256 of ORIGINAL dataset
with open("data/raw/train.csv", "rb") as f:
    data_sha256 = hashlib.sha256(f.read()).hexdigest()


# Load and clean dataset
df = pd.read_csv("data/raw/train.csv")
df = clean_train(df)


# Split data chronologically
X_train, X_valid, y_train, y_valid = split_data(
    df,
    test_size=params["data"]["test_size"]
)


# Configure MLflow
mlflow.set_tracking_uri(params["tracking"]["uri"])
mlflow.set_experiment(params["experiment"]["name"])


results = []


# Train all configurations
for config in params["model"]["configurations"]:

    config = config.copy()
    config_name = config.pop("name")

    print(f"\nTraining: {config_name}")

    pipeline = build_pipeline(
        config,
        random_state=params["data"]["random_state"]
    )

    with mlflow.start_run(run_name=config_name) as run:

        # Log parameters
        mlflow.log_param("model_type", params["model"]["type"])
        mlflow.log_param("configuration", config_name)
        mlflow.log_params(config)

        # Log original dataset hash
        mlflow.log_param("data_sha256", data_sha256)

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate model
        predictions = pipeline.predict(X_valid)
        mae = mean_absolute_error(y_valid, predictions)

        # Log metric
        mlflow.log_metric("validation_mae", mae)

        # Log trained model
        mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        results.append({
            "run_id": run.info.run_id,
            "configuration": config_name,
            "mae": mae
        })

        print("Validation MAE:", mae)


# Select best model
best_result = min(results, key=lambda x: x["mae"])

print("\nBest configuration:", best_result["configuration"])
print("Best MAE:", best_result["mae"])
print("Best run ID:", best_result["run_id"])


# Register best model
model_name = params["tracking"]["registered_model_name"]

model_uri = f"runs:/{best_result['run_id']}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)


# Assign champion alias
client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered_model.version
)

print("\nRegistered model:", model_name)
print("Model version:", registered_model.version)
print("Champion alias assigned successfully.")
