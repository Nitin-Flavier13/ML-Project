import os
import sys 
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation

# for defining variables
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion")
        try:
            # -- will be updating
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  

            logging.info('Train Test Split initiated')
            train_set,test_set = train_test_split(df,test_size=0.33,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    num_features = ['writing score','reading score']
    cat_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

    obj = DataTransformation(target_column="math score",num_features=num_features,cat_features=cat_features)
    train_arr,test_arr,preproccesor_path = obj.initiate_data_tranformation(train_path=train_data_path,test_path=test_data_path)

    modelTrainer = ModelTrainer()
    best_model_score,best_tuned_score = modelTrainer.initiate_model_trainer(train_arr,test_arr)
    print("Best Model score: ",best_model_score)
    print("Best Tuned Model score: ",best_model_score)
