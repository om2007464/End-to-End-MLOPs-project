import logging
import numpy as np
from prefect import task, flow
from src.pipeline.training_pipeline import TrainingPipeline

# Initialize the training pipeline
trainingPipeline = TrainingPipeline()

logging.basicConfig(level=logging.INFO)

# Define the tasks using the @task decorator
@task
def start_data_ingestion():
    trainDataPath, testDataPath = trainingPipeline.start_data_ingestion()
    if trainDataPath is None or testDataPath is None:
        logging.error("Data ingestion failed, invalid paths returned.")
        raise Exception("Data ingestion failed")
    return trainDataPath, testDataPath


@task
def transform_data(trainDataPath, testDataPath):
    logging.info("Starting data transformation...")
    trainData, testData = trainingPipeline.start_data_transformation(
        trainPath=trainDataPath, 
        testPath=testDataPath
    )
    # If the transformation is supposed to return numpy arrays, no need to convert to list
    # If you need to, convert it back here to ensure it's in the correct format
    trainData = trainData.tolist() if isinstance(trainData, np.ndarray) else trainData
    testData = testData.tolist() if isinstance(testData, np.ndarray) else testData
    return trainData, testData

@task
def train_model(trainData, testData):
    logging.info("Starting model training...")
    trainData = np.array(trainData)  # Ensure it's in numpy format for model training
    testData = np.array(testData)
    trainingPipeline.start_model_training(trainData=trainData, testData=testData)
    logging.info("Model training completed.")

@task
def evaluate_model(trainData, testData):
    logging.info("Starting model evaluation...")
    trainData = np.array(trainData)  # Ensure it's in numpy format for evaluation
    testData = np.array(testData)
    trainingPipeline.eval_model_metrics(trainData=trainData, testData=testData)
    logging.info("Model evaluation completed.")

# Define the flow using @flow decorator
@flow
def gemstone_training_pipeline():
    # Define the flow's task dependencies
    trainDataPath, testDataPath = start_data_ingestion()
    trainData, testData = transform_data(trainDataPath, testDataPath)
    train_model(trainData, testData)
    evaluate_model(trainData, testData)

# Run the flow
if __name__ == "__main__":
    gemstone_training_pipeline()
