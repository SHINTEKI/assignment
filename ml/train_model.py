import os
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

# Load data
base_dir = os.path.dirname(os.path.dirname(__file__)) 
file_path = os.path.join(base_dir, "data", "Real estate.csv")
df = pd.read_csv(file_path)

# Feature Engineering

# Extract year and day fraction from transaction date
df['Year'] = df['X1 transaction date'].astype(int)
df['YearFraction'] = df['X1 transaction date'] - df['Year']
df['DayOfYear'] = (df['YearFraction'] * 365).round().astype(int)

# Infer month from day of year
df['Month'] = df['DayOfYear'] // 30
df['Month'] += (df['DayOfYear'] % 30 > 0).astype(int)
df['Month'] = df['Month'].clip(upper=12)

# Log-transform MRT distance to reduce skew
df['log_distance_to_MRT'] = np.log1p(df['X3 distance to the nearest MRT station'])

# Calculate distance to city center (mean coordinates)
center_lat = df['X5 latitude'].mean()
center_lon = df['X6 longitude'].mean()
df['dist_to_center'] = np.sqrt((df['X5 latitude'] - center_lat) ** 2 + (df['X6 longitude'] - center_lon) ** 2)

# Target variable
target = 'Y house price of unit area'

# Feature lists
numeric_features = [
    'X2 house age',
    'log_distance_to_MRT',
    'X4 number of convenience stores',
    'dist_to_center'
]
categorical_features = ['Month']

X = df[numeric_features + categorical_features]
y = df[target]

# Preprocessing: scale numerics, one-hot encode categoricals
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# ElasticNet with CV
model = Pipeline([
    ('prep', preprocessor),
    ('reg', ElasticNetCV(l1_ratio=0.5, cv=5))
])

# Fit the model
model.fit(X, y)

# Save trained model
joblib.dump(model, "predictor.joblib")
