import os
import sys 
from dataclasses import dataclass 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting train and test data')

            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            ) 

            models = {
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "XGBoostRegressor": XGBRegressor(),
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "K-Neighbors-Regressor": KNeighborsRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostRegressor": GradientBoostingRegressor()
            }  

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_name,best_model_score = sorted(model_report.items(),key=lambda x:x[1],reverse=True)[0]

            best_model = models[best_model_name]

            if best_model_score <= 0.6:
                raise CustomException("Best Model not found",sys)
            logging.info(f'Best Model found {best_model_name} with {best_model_score} accuracy')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Saved the Best Model')

            return best_model_score


        except Exception as e:
            raise CustomException(e,sys)

    
