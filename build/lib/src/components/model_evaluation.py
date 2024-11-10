import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
import pandas as pd 
from src.logger.logging import logging
import pickle
from src.utils.utils import load_object
from src.exception.exception import CustomException
import mlflow 
import mlflow.sklearn
from urllib.parse import urlparse 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import dagshub
from src.components.data_ingestion import DataIngestionConfig



@dataclass
class ModelEvaluationConfig:
    pass 


class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation has started")

        
    def eval_metrics(self,y_true, preds):
        rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true=y_true, y_pred=preds)
        logging.info("Evaluating metrics")
        return rmse, mae, r2 
    
    def InitiateModelEvaluation(self, trainData, testData):
        try: 
            X_test, y_test = trainData[:,:-1], testData[:, -1]
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            dagshub.init(repo_owner='peniel18', repo_name='Gemstone-Price-Prediction', mlflow=True)

            MLFLOW_TRACKING_URI = "https://dagshub.com/peniel18/Gemstone-Price-Prediction.mlflow"
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            

            with mlflow.start_run(): 
                predictions = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, preds=predictions)
                # log metrics 
                mlflow.log_metric("rmse", rmse) 
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)   
                # log artifacts 
                model_dir = DataIngestionConfig.artifacts_dir
                mlflow.log_artifact(model_dir) 
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e: 
            logging.info("Error Occured in model_evaluation.py")
            raise CustomException(e, sys)
    
