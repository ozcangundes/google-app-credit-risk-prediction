from flask import Flask, request
import joblib
import pandas as pd
import numpy as np
import pipe_funcs
from pipe_funcs import CategoricalImputer,CategoricalEncoder,BoxcoxTransformer
import os
import logging




app = Flask(__name__)


model=joblib.load("model.pkl")

pipeline=joblib.load("pipeline.pkl")

    

@app.route('/api_predict', methods=["GET", "POST"])
def api_predict(): 

    
    if request.method == "GET":
        message='Please send Post Request.\nSample request:\n {"Age":22,\n"Sex":"female",\n"Job":2,\n "Housing":"own",\n "Saving accounts":"little",\n "Checking account":"moderate",\n "Credit amount":5951,\n "Duration":48,\n "Purpose":"radio/TV"}\n'
        return message
    elif request.method == "POST":
        data = request.get_json()
        
        inp=pd.json_normalize(data)
        inp=pipeline.transform(inp)        
        prediction = model.predict(inp)

        if prediction[0]==0:
            output="Credit risk is BAD!"
        else:
            output="Credit risk is GOOD!"
        
        return str(output)



if __name__=="__main__":


    app.run()