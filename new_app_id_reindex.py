# Data Handling
import joblib
import csv

import numpy as np
import pandas as pd

# S3
import boto3
from io import BytesIO
s3 = boto3.client('s3')
bucket_name = 'fastapimodels2' # to be replaced with bucket name

# Server
from fastapi import FastAPI

from pydantic import BaseModel, Field

app = FastAPI()

# Initialize files
def read_s3_joblib_file(key):
    with BytesIO() as data:
        s3.download_fileobj(Fileobj=data, Bucket=bucket_name, Key=key)
        data.seek(0)
        return joblib.load(data)

def read_s3_csv_file(key):
    with BytesIO() as data:
        s3.download_fileobj(Fileobj=data, Bucket=bucket_name, Key=key)
        data.seek(0)
        return pd.read_csv(data)


clf = read_s3_joblib_file('random_forest_best.joblib')
df_test = read_s3_csv_file('test_ID_reindex.csv')

# Class which describes a single utilisateur
class Parametres(BaseModel):

    SK_ID_CURR: int = Field(title = "ID du prêt", ge=100001, le=456250)
    
@app.get("/status")
def get_status():
    """Get status of messaging server."""
    return ({"status":  "running"})
               
@app.post("/predict")
def predict(datas: Parametres ):

    # Extract data in correct order
    datas_dict = datas.dict()

    df = pd.DataFrame([datas_dict])
    print(df)
    
    df_2=df_test
    print(df_2)

    df_3 = df.merge(right=df_2, on = 'SK_ID_CURR', how = 'inner')
    print(df_3)
    

    df_3=df_3.set_index('SK_ID_CURR')

    # Create and return prediction

    threshold = 0.680000
    prediction = (clf.predict_proba(df_3)[:, 1] > threshold).astype('float')

    #prediction = clf.predict(df_3)
    #prediction = prediction.tolist()
    probability = clf.predict_proba(df_3).max()


    return {'prediction': prediction[0],
           'probability': probability,
            }
    
