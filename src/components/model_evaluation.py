import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object
from src.components.data_ingestion import DataIngestionConfig

@dataclass
class ModelEvaluationConfig:
    """Configuration class for model evaluation."""
    pass


class ModelEvaluation:
    def __init__(self):
        logging.info("Model Evaluation has started")

    def eval_metrics(self, y_true: np.ndarray, preds: np.ndarray) -> tuple:
        """Calculate and return evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        logging.info("Evaluating metrics: RMSE, MAE, R2")
        return rmse, mae, r2

    def _load_model(self, model_path: str):
        """Load the model from the given path."""
        try:
            model = load_object(model_path)
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {model_path}")
            raise CustomException(e, sys)

    def _log_metrics_to_mlflow(self, rmse: float, mae: float, r2: float):
        """Log metrics to MLflow."""
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        logging.info("Metrics logged to MLflow")

    def _log_model_artifacts(self, model: object):
        """Log model and artifacts to MLflow."""
        model_dir = DataIngestionConfig.artifacts_dir
        mlflow.log_artifact(model_dir)
        tracking_url = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
        else:
            mlflow.sklearn.log_model(model, "model")
        logging.info("Model artifacts logged to MLflow")

    def initiate_model_evaluation(self, train_data: np.ndarray, test_data: np.ndarray):
        """Evaluate the model performance using test data."""
        try:
            # Split test data into features and target
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = self._load_model(model_path)

            # Initialize MLflow for local tracking
            mlflow.set_tracking_uri(r"mlruns")  # Set the local directory for tracking
            with mlflow.start_run():
                # Get predictions from the model
                predictions = model.predict(X_test)

                # Evaluate metrics
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                # Log metrics to MLflow
                self._log_metrics_to_mlflow(rmse, mae, r2)

                # Log model and artifacts to MLflow
                self._log_model_artifacts(model)

        except Exception as e:
            logging.error("Error occurred during model evaluation")
            raise CustomException(e, sys)