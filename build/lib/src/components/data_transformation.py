import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
import pandas as pd 
from typing import Tuple
from src.logger.logging import logging
from src.exception.exception import CustomException
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


from src.utils.utils import save_object 



@dataclass
class DataTransformationConfig:
    current_dir = os.getcwd()
    # path for the preprocessor in the artifacts folder 
    preprocessorObjPath = os.path.join(current_dir, "..", "..", "artifacts", "preprocessor.pkl")
    preprocessorName = "preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()
        

    def getDataTransformation(self):
        try: 
            logging.info("Data Transformation Started")
            # catogorical columns and numerical columns 
            categorical_columns = ["cut", "color", "clarity"]
            numerical_columns = ["carat", "depth", "table", "x", "y", "z"]

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Pipeline Initaited")

            # numerical pipeline 
            numPipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")), 
                    ("Scaler", StandardScaler())
                ]
            )

            catPipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")), 
                    ("OrdinalEconder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])), 
                    ("Scaler", StandardScaler())
                ]
            )

            # preprocessing 
            preprocessor = ColumnTransformer(
                [
                    ("numPipeline", numPipeline, numerical_columns), 
                    ("catPipeline", catPipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e: 
            logging.info("Exception occurred in the getDataTransformation")
            raise CustomException(e, sys)
    

    def InitialDataTransformation(self, train_path, test_path):
        try: 
            logging.info("IntialDataTransformation function is Initiated")
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path)
            
            preprocessor = self.getDataTransformation()
            targetColumn = "price"
            dropColumns = [targetColumn, "id"]
            # Get Training Features and Target 
            targetTrain = train_df[targetColumn]
            featuresTrain = train_df.drop(columns=dropColumns, axis=1)
            
            # Get Testing Features and Target 
            targetTest = test_df[targetColumn] 
            featuresTest = test_df.drop(columns=dropColumns, axis=1)
              
            # preprocessing 
            logging.info("Preprocessing Both Training and testing data")
            featuresTrainPreData = preprocessor.fit_transform(featuresTrain)
            featuresTestPreData = preprocessor.transform(featuresTest)
            # combining data 
            TrainData = np.c_[featuresTrainPreData, np.array(targetTrain)]
            TestData = np.c_[featuresTestPreData, np.array(targetTest)]
            
            logging.info("Save the Preprocessor Object as a pickle file")
            save_object(
                file_path=self.DataTransformationConfig.preprocessorObjPath, 
                obj=preprocessor 
                )

            return (TrainData, TestData)

        except Exception as e: 
            logging.info("Exception occured in the IntialDataTransformation")
            raise CustomException(e, sys)