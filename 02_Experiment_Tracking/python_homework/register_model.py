import os
import pickle
import click
import mlflow
import random

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model artifact
        mlflow.sklearn.log_model(rf, artifact_path="model")


def generate_random_params():
    return {
        "max_depth": random.choice([5, 10, 15, 20]),
        "n_estimators": random.choice([50, 100, 150]),
        "min_samples_split": random.choice([2, 5, 10]),
        "min_samples_leaf": random.choice([1, 2, 4]),
        "random_state": 42
    }


@click.command()
@click.option("--data_path", default="./output", help="Path to pickled train/val/test sets")
@click.option("--n_runs", default=5, type=int, help="Number of random runs to evaluate")
def run_experiment(data_path: str, n_runs: int):

    for _ in range(n_runs):
        params = generate_random_params()
        train_and_log_model(data_path=data_path, params=params)

    # Find best run
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="RandomForestBestModel")


if __name__ == "__main__":
    run_experiment()
# This script trains a Random Forest model with random hyperparameters,
# logs the results, and registers the best model found during the runs.
# It uses MLflow for experiment tracking and model management.
# The script can be run from the command line with the specified data path and number of runs.
# Example usage:
# python register_model.py --data_path ./output --n_runs 5