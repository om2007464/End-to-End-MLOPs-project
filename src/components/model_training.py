import os
import sys
from dataclasses import dataclass
from comet_ml import Experiment  # Import comet_ml for experiment tracking
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.logger.logging import logging
from src.utils.utils import save_object, evaluate_model
from src.exception.exception import CustomException

# Configuration for storing the trained model
@dataclass
class ModelTrainerConfig:
    current_dir = os.getcwd()  # Get current working directory
    trainedModelPath = os.path.join(current_dir, "artifacts", "model.pkl")  # Path to save the trained model
    modelName = "model.pkl"  

# Class to handle model training
class ModelTrainer:
    def __init__(self):
        # Initialize configuration for model training
        self.ModelTrainerConfig = ModelTrainerConfig()
        # Initialize a Comet experiment (replace "your-api-key" with your actual Comet API key)
        self.experiment = Experiment(
            api_key="Add your API key here",
            project_name="gemstone-price-prediction",  # Name of your project
            workspace="Add your work space name here "  # Your Comet workspace
        )

    # Function to start the model training process
    def InitiateModelTraining(self, trainData, testData):
        try: 
            logging.info("Splitting Data for Training")
            # Split the training and testing data into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                trainData[:, :-1],  # Features from trainData (all columns except last)
                trainData[:, -1],   # Target from trainData (last column)
                testData[:, :-1],   # Features from testData (all columns except last)
                testData[:, -1]     # Target from testData (last column)
            )

            # Log dataset parameters
            self.experiment.log_dataset_hash(trainData)
            self.experiment.log_parameters({
                "train_data_shape": trainData.shape,
                "test_data_shape": testData.shape
            })

            # Define a dictionary of models to be trained
            models = {
                "LinearRegression" : LinearRegression(), 
                "Lasso" : Lasso(), 
                "Ridge" : Ridge(), 
                "Elasticnet": ElasticNet()
            }

            # Evaluate each model using the evaluate_model utility function
            models_report: dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                                 X_test=X_test, y_test=y_test, models=models)
            
            print(models_report)
            logging.info(f"Models Report : {models_report}")
            
            # Log model metrics to Comet
            for model_name, score in models_report.items():
                self.experiment.log_metric(f"{model_name}_R2_score", score)

            # Get the best model score and corresponding model name
            best_model_score = max(models_report.values())  # Get highest R2 score
            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            # Select the best model from the models dictionary
            best_model = models[best_model_name]
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            # Log best model details to Comet
            self.experiment.log_parameter("best_model_name", best_model_name)
            self.experiment.log_metric("best_model_score", best_model_score)

            # Save the best model to a file (as a pickle object)
            save_object(
                file_path=self.ModelTrainerConfig.trainedModelPath,  
                obj=best_model
            )

            # Log model to Comet (optional)
            self.experiment.log_model(best_model_name, self.ModelTrainerConfig.trainedModelPath)

        # Handle exceptions during the model training process
        except Exception as e: 
            logging.error("Exception occurred during model training")
            self.experiment.log_exception(e)
            raise CustomException(e, sys)
        finally:
            # End the Comet experiment
            self.experiment.end()
