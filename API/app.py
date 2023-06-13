import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
import joblib
import pickle
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Create the app object
app = FastAPI()

# Load the scaler
scaler = joblib.load('/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/models/scaler.pkl')

# Load the saved model
with open('/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a class that describes the input
class SmokeStatus(BaseModel):
    age: int
    height: int
    weight: int
    eyesight_left: float
    eyesight_right: float
    hearing_left: int
    hearing_right: int
    systolic: int
    fasting_blood_sugar: int
    Cholesterol: int
    triglyceride: int
    HDL: int
    hemoglobin: float
    urine_protein: int
    serum_creatinine: float
    AST: int
    ALT: int
    Gtp: int
    dental_caries: int

# Define the routes
@app.get('/')
def index():
    return {'message': 'Hello, Everyone!'}

@app.get('/Welcome/{name}')
def get_name(name: str):
    return {'Welcome to my ML model': name}

@app.post('/predict')
def predict_smoke(data: SmokeStatus):
    data = data.dict()
    # Scale the data
    data = scaler.transform([[data[var] for var in data]])
    # Make prediction
    prediction = model.predict(data)
    # Convert prediction to a native Python data type
    prediction = prediction.tolist()
    return json.dumps(prediction, cls=NumpyJSONEncoder)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
