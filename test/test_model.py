import joblib
import numpy as np
import pandas as pd
import os

def test_model_file_exists():
    assert os.path.exists("predictor.joblib"), "Model file not found!"

def test_model_load_and_predict():
    model = joblib.load("predictor.joblib")
    assert model is not None, "Failed to load model."

    # Sample input:
    # Features (in order): house_age, log_distance_to_MRT, convenience_stores, dist_to_center, month
    sample_input = pd.DataFrame([{
        "X2 house age": 20,
        "log_distance_to_MRT": np.log1p(300),
        "X4 number of convenience stores": 5,
        "dist_to_center": 0.02,
        "Month": 3
    }])

    pred = model.predict(sample_input)
    assert pred.shape == (1,), "Prediction output shape mismatch."
    assert pred[0] > 0, "Prediction is not positive — unexpected output."
