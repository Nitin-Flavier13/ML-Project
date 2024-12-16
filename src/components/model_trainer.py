import os
import sys 
from dataclasses import dataclass 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model, get_tuned_model

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

            params= {
                # so we will take AdaBoost and GradientBoost for hyper parameter tunning
                'Lasso' : {
                    "alpha": [0.1, 0.01, 0.001, 1, 10],  # Regularization strength
                    "max_iter": [1000, 5000, 10000],  # Number of iterations for optimization
                    "selection": ['cyclic', 'random']  # Optimization technique
                },

                'Ridge' : {
                    "alpha": [0.1, 0.01, 0.001, 1, 10],  # Regularization strength
                    "solver": ['auto', 'lsqr', 'saga', 'cholesky'],  # Optimization algorithm
                    "max_iter": [1000, 5000, 10000]  # Number of iterations for optimization
                },

                'AdaBoostRegressor' : {
                    "n_estimators": [50,55,60,65,70,80],
                    "loss": ['linear','square','exponential']
                },

                'GradientBoostRegressor' : {
                    "loss": ['squared_error','huber','absolute_error'],
                    "criterion": ['friedman_mse','squared_error','mse'],
                    "min_samples_split": [2,8,15,20],
                    "n_estimators": [100,200,500,1000],
                    "max_depth": [5,8,15,None,10],
                    "learning_rate": [0.1,0.01,0.02,0.2]
                }
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_name,best_model_score = sorted(model_report.items(),key=lambda x:x[1],reverse=True)[0]

            if best_model_score <= 0.6:
                raise CustomException("Best Model not found",sys)
            logging.info(f'Best Model found {best_model_name} with {best_model_score} accuracy')

            # hyper-parameter tunning on 4 best models
            
            logging.info('Hyper-Parameter tuning started')

            best_tuned_model = get_tuned_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,params=params,models=models)
           
            logging.info(f'Best tuned model is {best_tuned_model[0]} with score of {best_tuned_model[1][0]}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_tuned_model[1][1]
            )
            logging.info('Saved the Best Tuned Model')

            return (best_model_score,best_tuned_model[1][0])


        except Exception as e:
            raise CustomException(e,sys)

    
