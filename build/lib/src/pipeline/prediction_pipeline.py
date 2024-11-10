import os 
import sys 
import pandas as pd 
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object


class PredictionPipeline: 
    def __init__(self) -> None:
        pass 

    def predict(self, features):
        try: 
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            # transform the features 
            TransformedFeatures = preprocessor.transform(features)
            y_hat = model.predict(TransformedFeatures)
            return y_hat 

            
        except Exception as e: 
            logging.info()
            CustomException(e, sys)


class CustomData: 
    def __init__(self, 
                 carat: float, 
                 depth: float, 
                 table: float, 
                 x: float, 
                 y: float, 
                 z: float, 
                 cut: str, 
                 color: str, 
                 clarity: str ):
        self.carat = carat 
        self.depth = depth
        self.table = table
        self.x = x 
        self.y = y 
        self.z = z 
        self.cut = cut 
        self.color = color 
        self.clarity = clarity 

    def getDataAsDataFrame(self):
        try: 
            custom_data_inputs = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_inputs)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e: 
            logging.info('Exception Occured in prediction pipeline')
            CustomException(e, sys)


if __name__ == "__main__":
    PredictionPipeline().predict()