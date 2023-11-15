from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/models/model.pkl")


def make_prediction(data: pd.DataFrame) -> List[int]:
    # Make predictions using the loaded model
    predictions = model.predict(data)
    return predictions.tolist()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a Pandas DataFrame
        data = pd.read_csv(file.file)

        # Make predictions
        predictions = make_prediction(data)

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}",
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
