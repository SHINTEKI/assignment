from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import math
import os

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "predictor.joblib")
model = joblib.load(model_path)

# Define input schema
class HouseFeatures(BaseModel):
    transaction_date: float  
    house_age: float
    distance_to_mrt: float
    convenience_stores: int
    latitude: float
    longitude: float

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    logger.info(f"Input received: {features.dict()}")

    try:
        # Feature engineering
        year = int(features.transaction_date)
        year_fraction = features.transaction_date - year
        day_of_year = int(round(year_fraction * 365))
        month = day_of_year // 30 + (1 if day_of_year % 30 > 0 else 0)
        month = min(month, 12)
        log_mrt_distance = math.log1p(features.distance_to_mrt)
        center_lat = 24.969030072463767  # calculated from training set
        center_lon = 121.53336108695655  # calculated from training set
        dist_to_center = math.sqrt((features.latitude - center_lat)**2 + (features.longitude - center_lon)**2)

        # Build input DataFrame
        X_input = pd.DataFrame([{
            "X2 house age": features.house_age,
            "log_distance_to_MRT": log_mrt_distance,
            "X4 number of convenience stores": features.convenience_stores,
            "dist_to_center": dist_to_center,
            "Month": month
        }])

        # Predict
        prediction = model.predict(X_input)[0]
        logger.info(f"Prediction made: {prediction}")
        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")

