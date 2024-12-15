import os
import sys 
import numpy as np
import pandas as pd

from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass 
from src.exception import CustomException

from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self,target_column,num_features,cat_features):
        self.cat_features = cat_features
        self.num_features = num_features
        self.target_column = target_column
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # num_features = ['writing score','reading score']
            # cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),   # handling missing values
                    ("scaler",StandardScaler())                     # scaling values
                ]
            )
            logging.info('Numerical Column Scaling Completed')

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
                ]
            )

            logging.info('Categorical Column Encoding Completed')

            preproccesor = ColumnTransformer(transformers=[
                ("Numerical_Transformer",num_pipeline,self.num_features),
                ("Categorical_Transformer",cat_pipeline,self.cat_features)
            ])

            logging.info('data transformation object made successfully')

            return preproccesor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            test_df = pd.read_csv(test_path)
            train_df = pd.read_csv(train_path)
            logging.info('data transformation test & train data read')

            preproccesor = self.get_data_transformer_object()
            
            target_feature_train_df = train_df[self.target_column]
            input_feature_train_df = train_df.drop(columns=[self.target_column],axis=1)

            target_feature_test_df = test_df[self.target_column]
            input_feature_test_df = test_df.drop(columns=[self.target_column],axis=1)

            logging.info('Applying Tranformation on train and test data')

            input_feature_train_arr = preproccesor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproccesor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('Returning scaled data')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preproccesor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
    

 