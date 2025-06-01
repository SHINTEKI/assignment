# Overview of project

I developed a simple machine learning microservice to predict real estate unit prices using an ElasticNet regression model trained on housing data. This service is built using FastAPI and scikit-learn, and supports training, model testing, and deployment.

---

## Solution Design

- Feature engineering:
  - Extract `month` from transaction date and use as a categorical feature, allowing the model to account for seasonal effects (e.g. peak vs. slack months)
  - Log transforms `distance to MRT` to solve its severe right-skewed distribution
  - Compute `distance to the city center` using the mean of all `latitude` and `longitude` values, making raw geographic coordinates more interpretable and useful
- Train with `ElasticNetCV` regression, which performs automatic cross-validation to select the best regularization parameters and helps prevent overfitting
- REST API with `/predict` and `/health` endpoints
- Unit tests for model validity

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python ml/train_model.py
```

This will generate `predictor.joblib` in the root folder. Please also copy the newest version to the app folder to utilize it as the base model for prediction.

### 3. Run the API server

```bash
uvicorn app.main:app --reload
```

- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Request Body:**
```json
{
  "transaction_date": 2013.25,
  "house_age": 20,
  "distance_to_mrt": 300,
  "convenience_stores": 5,
  "latitude": 24.965,
  "longitude": 121.540
}
```
**Response:**
```json
{
  "predicted_price": 45.23
}
```

### 4. Run tests

```bash
pytest test/
```

---

## Dependencies

```
fastapi==0.115.12
joblib==1.5.1
numpy==2.2.6
pandas==2.2.3
pydantic==2.11.5
scikit_learn==1.6.1
pytest==8.3.5
uvicorn==0.34.3 
```