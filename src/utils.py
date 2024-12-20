import os
import sys 
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for modelName,model in models.items():
            model.fit(X_train,y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[modelName] = test_model_score

        return report            
    except Exception as e:
        raise CustomException(e,sys)

def get_tuned_model(X_train,y_train,X_test,y_test,params,models):
    try:
        report = {}
        for modelName,param in params.items():

            gcv = GridSearchCV(estimator=models[modelName],param_grid=param,cv=5,refit=True,verbose=2)
            gcv.fit(X_train,y_train)

            y_test_pred = gcv.predict(X_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[modelName] = [test_model_score,gcv]

        best_tuned_model = sorted(report.items(),key=lambda x:x[1][0],reverse=True)[0]

        return best_tuned_model

    except Exception as e:
        raise CustomException(e,sys)

