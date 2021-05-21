from flask import Flask, request
import joblib
import pandas as pd
import numpy as np
from pipe_funcs import *


app = Flask(__name__)

        



@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    from sklearn.base import BaseEstimator, TransformerMixin

    class CategoricalImputer(BaseEstimator, TransformerMixin):
        """Categorical data missing value imputer."""
    
        def __init__(self, variables=None) -> None:
            if not isinstance(variables, list):
                self.variables = [variables]
            else:
                self.variables = variables
    
        def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
            """Fit statement to accomodate the sklearn pipeline."""
    
            return self
    
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """Apply the transforms to the dataframe."""
    
            X = X.copy()
            for feature in self.variables:
                X[feature]=X[feature].replace("",np.nan)
                X[feature] = X[feature].fillna("Missing")
    
            return X
        
    class CategoricalEncoder(BaseEstimator, TransformerMixin):
        """String to numbers categorical encoder."""
    
        def __init__(self, variables=None):
            if not isinstance(variables, list):
                self.variables = [variables]
            else:
                self.variables = variables
    
        def fit(self, X, y=None):
            temp = X.copy()
            
            # persist transforming dictionary
            self.encoder_dict_ = {}
    
            for var in self.variables:
                t = temp.groupby([var])["Age"].count().sort_values(ascending=True).index
                self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
    
            return self
    
        def transform(self, X):
            # encode labels
            X = X.copy()
            for feature in self.variables:
                X[feature] = X[feature].map(self.encoder_dict_[feature])
                    # check if transformer introduces NaN
            if X[self.variables].isnull().any().any():
                null_counts = X[self.variables].isnull().any()
                vars_ = {
                    key: value for (key, value) in null_counts.items() if value is True
                }
                print(
                    f"Categorical encoder has introduced NaN when "
                    f"transforming categorical variables: {vars_.keys()}"
                )
    
            return X
    
    class BoxcoxTransformer(BaseEstimator, TransformerMixin):
    
        def __init__(self, variables=None):
            if not isinstance(variables, list):
                self.variables = [variables]
            else:
                self.variables = variables
    
        def fit(self, X, y=None):
            # to accomodate the pipeline
            return self
    
        def transform(self, X):
            X = X.copy()
    
            for feature in self.variables:
                #X[feature] = special.boxcox1p(X[feature],0)
                X[feature] = np.log(X[feature]+1)
                
    
            return X
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    #config features
    CATEGORICAL_VARS_WITH_NA = ["Saving accounts","Checking account"]
    CATEGORICAL_VARS=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    NUMERICAL_VARS_FOR_BOXCOX=["Age","Credit amount","Duration"]
    
    pipeline=Pipeline([
        (
            "categorical_imputer", CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "categorical_encoder", CategoricalEncoder(variables=CATEGORICAL_VARS),
        ),
        (
            "boxcox_transformer", BoxcoxTransformer(variables=NUMERICAL_VARS_FOR_BOXCOX),
        ),
        (   "scaler", MinMaxScaler(),
        )
    ])

    model=joblib.load("model.pkl")
    pipeline=joblib.load("pipeline.pkl")
    
    
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
    model=joblib.load("model.pkl")
    pipeline=joblib.load("pipeline.pkl")
    app.run()