import numpy as np
import pandas as pd

from flask import Flask,request,render_template

from sklearn.preprocessing import StandardScaler 
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application 

@app.route('/',methods=['POST','GET'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data_obj = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )

        dataframe = data_obj.get_data_as_dataFrame()
        logging.info('got the input data from the form')

        predict_obj = PredictPipeline()
        answer = predict_obj.predict_data(dataframe)

        return render_template('home.html',result=answer[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)



