from flask import Flask, request
import joblib
import pandas as pd
import numpy as np
from pipe_funcs import *


app = Flask(__name__)

        
model=joblib.load("model.pkl")
pipeline=joblib.load("pipeline.pkl")


@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():

    

    
    if request.method == "GET":
        return "Please send Post Request"
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