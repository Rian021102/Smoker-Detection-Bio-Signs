from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the machine learning model using joblib
model_path = "/Users/rianrachmanto/pypro/project/smoker-detection/models/model.joblib"
model = joblib.load(model_path)

# Load the scaler using joblib
scaler_path = "/Users/RianRachmanto/pypro/project/smoker-detection/models/scaler.joblib"
scaler = joblib.load(scaler_path)

# Create a Pydantic model for the request body
class Item(BaseModel):
    data: str

# Define a FastAPI route
@app.post("/predict")
def predict(item: Item):
    # Get the data from the request body
    data = item.data

    # Load the test data from the CSV file
    test_data_path = "/Users/RianRachmanto/pypro/project/smoker-detection/data/raw/test-smoking.csv"
    test_data = pd.read_csv(test_data_path)

    # List of columns to delete
    columns_to_delete = ["column1", "column2", "column3"]

    # Delete the specified columns
    test_data = test_data.drop(columns=columns_to_delete)

    # Apply scaling to the data
    scaled_data = scaler.transform(test_data)

    # Preprocess the data if needed

    # Make predictions using the loaded model
    prediction = model.predict(scaled_data)

    # Return the prediction as the response
    return {"prediction": prediction.tolist()}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
