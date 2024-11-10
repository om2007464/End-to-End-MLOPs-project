import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
import pandas as pd 
from src.logger.logging import logging
from src.utils.utils import save_object, evaluate_model
from src.exception.exception import CustomException
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


@dataclass
class ModelTrainerConfig:
    current_dir = os.getcwd()
    trainedModelPath = os.path.join(current_dir, "artifacts", "model.pkl")
    modelName = "model.pk1"

class ModelTrainer:
    def __init__(self):
        self.ModelTrainerConfig = ModelTrainerConfig()

    def InitiateModelTraining(self, trainData, testData):
        try: 
            logging.info("Spliting Data for Training")
            X_train, y_train, X_test, y_test = (
                trainData[:, :-1], 
                trainData[:, -1], 
                testData[:, :-1], 
                testData[:, -1]
            )

            models = {
                "LinearRegression" : LinearRegression(), 
                "Lasso" : Lasso(), 
                "Ridge" : Ridge(), 
                "Elasticnet": ElasticNet()
            }

            models_report: dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                                 X_test=X_test, y_test=y_test, models=models)
            print(models_report)
            print('\n====================================================================================\n')
            logging.info(f"Models Report : {models_report}")
            # get the best model score and name
            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[
                                    list(models_report.values()).index(best_model_score)
                                ]
            best_model = models[best_model_name]
            print(f'Best Model : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.ModelTrainerConfig.trainedModelPath,  
                obj=best_model
            )
            

        except Exception as e: 
            logging.error("Exception occured at model training")
            raise CustomException(e, sys)
