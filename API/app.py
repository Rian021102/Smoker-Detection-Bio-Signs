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

# Load the saved model
with open('ml_process_demo/models/model.pkl', 'rb') as f:
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
    # Convert the input data to a dictionary and extract the values
    data_dict = data.dict()
    data_values = list(data_dict.values())

    # Convert the values to a 2D array
    input_data = np.array([data_values])

    # Make prediction
    prediction = model.predict(input_data)

    # Interpret the prediction
    if prediction == 0:
        smoker_status = "Non-Smoker"
    else:
        smoker_status = "Smoker"

    return {"smoker_status": smoker_status}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
