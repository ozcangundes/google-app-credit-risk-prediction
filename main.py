from flask import Flask, request
import joblib
import pandas as pd
import numpy as np
from pipe_funcs import CategoricalImputer,CategoricalEncoder,BoxcoxTransformer


app = Flask(__name__)


model=joblib.load("model.pkl")
pipeline=joblib.load("pipeline.pkl")
    
def predict(data,model=model,pipeline=pipeline):
    inp=pipeline.transform(data)
    prediction=model.predict(inp)
    return prediction
        

@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():

    
    if request.method == "GET":
        return "Please send Post Request"
    elif request.method == "POST":
        data = request.get_json()
        
        inp=pd.json_normalize(data)
                
        prediction = predict(inp)

        if prediction[0]==0:
            output="Credit risk is BAD!"
        else:
            output="Credit risk is GOOD!"
        
        return str(output)



if __name__=="__main__":
    model=joblib.load("model.pkl")
    pipeline=joblib.load("pipeline.pkl")
    app.run()