import datetime
from influxdb import InfluxDBClient
import math
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import pickle
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


mlflow.set_tracking_uri('http://training.itu.dk:5000/')
mlflow.sklearn.autolog()

with mlflow.start_run():
    
    dataset_time = datetime.datetime.now()
    mlflow.log_param("dataset_time", dataset_time)
    
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
    client.switch_database('orkney')

    def get_df(results):
        values = results.raw["series"][0]["values"]
        columns = results.raw["series"][0]["columns"]
        df = pd.DataFrame(values, columns=columns).set_index("time")
        #df.index = pd.to_datetime(df.index).astype(int)/ 10**9
        df.index = pd.to_datetime(df.index) # Convert to datetime-index
        return df

    # Get the last 90 days of power generation data
    generation = client.query(
        "SELECT * FROM Generation where time > now()-90d"
        ) # Query written in InfluxQL

    # Get the last 90 days of weather forecasts with the shortest lead time
    wind  = client.query(
        "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
        ) # Query written in InfluxQL

    gen_df = get_df(generation)
    wind_df = get_df(wind)
    new_gen_df = gen_df.groupby(pd.Grouper(freq= "1H")).mean()
    df = new_gen_df.join(wind_df, how='inner')
    df.dropna(inplace=True)

    # Creating test and train set
    n = len(df)
    train_df = df[:int(n*0.8)]
    test_df = df[int(n*0.8):]

    X = train_df[['Speed','Direction']]
    y = train_df['Total']

    X_test = test_df[['Speed','Direction',]]
    y_test = test_df['Total']

    def wind_to_vector(direction):
        angle = 0
        compass = { "E": 0, "N":math.pi/2, "W":math.pi, "S":math.pi*1.5}
        for c in direction.Direction:
            angle += compass[c]
        angle /= len(direction.Direction)
        x = round(math.cos(angle),2)
        y = round(math.sin(angle),2)
        return x, y

    class Direction_Transformer(BaseEstimator, TransformerMixin):
        def __init__(self):       
            pass
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y = None):
            X_ = pd.DataFrame(X['Direction'].copy())
            X_ = X_.apply(wind_to_vector, axis=1, result_type='expand')
            X_ = X_.rename(columns={0:'X', 1:'Y'})
            return X_


    col_transformer = ColumnTransformer([('direction', Direction_Transformer(), ['Direction']),
                                        ('poly', PolynomialFeatures(), ['Speed']),
                                        ])
    pipe = Pipeline([('transformer', col_transformer),
                    ('std_scaler', StandardScaler()),
                    ('linear', LinearRegression())])

    param_grid = {"transformer__poly__degree": [1,2,3,4,5],
                    "transformer__poly__interaction_only": [False]}

    #TimeSeriesSplit of 72 hours at a time (24 3-hour long datapoints) to emulate predicting 72hs into the future
    gsc = GridSearchCV(pipe, param_grid = param_grid, scoring='neg_mean_squared_error',
                cv=TimeSeriesSplit(n_splits=len(X)//24).split(X), verbose=4, n_jobs=-1, refit=True, return_train_score=True)

    gsc.fit(X, y)
