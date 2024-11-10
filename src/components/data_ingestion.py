import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Tuple
from pymongo import MongoClient
from src.logger.logging import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    current_dir = os.getcwd()
    artifacts_dir = os.path.join(current_dir, "..", "..", "artifacts")

    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.mongo_client = MongoClient("mongodb+srv://Spiderman:Spiderman@cluster0.ctt8i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.db = self.mongo_client['Spiderman']
        self.collection = self.db['raw_data']

    def InitialDataIngestion(self) -> Tuple[str, str]:
        logging.info("Data Ingestion Started")
        try:
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Fetch data from MongoDB
            raw_data = list(self.collection.find())
            if not raw_data:
                raise CustomException("No data found in MongoDB", sys)

            # Convert MongoDB data to DataFrame
            data = pd.DataFrame(raw_data)
            logging.info("Data fetched from MongoDB successfully")

            # Save raw data to CSV
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Train-test split
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save train and test data as CSV
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    # Start data ingestion
    obj.InitialDataIngestion()
