import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "tesla_deliveries_dataset_2015_2025.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# -----------------------------------------------------------------------------
# 2. Train Random Forest Model
# -----------------------------------------------------------------------------
print("Training Random Forest Model...")
# Drop target and CO2_Saved_tons (leakage), keep Production_Units for scenario simulator
# We also drop 'Date' if it exists, but the raw data has 'Year' and 'Month'
cols_to_drop = ['Estimated_Deliveries', 'CO2_Saved_tons', 'Date']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['Estimated_Deliveries']

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15))
])

pipeline.fit(X, y)

rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
joblib.dump(pipeline, rf_model_path)
print(f"Random Forest Model saved to {rf_model_path}")

# -----------------------------------------------------------------------------
# 3. Train Prophet Model
# -----------------------------------------------------------------------------
print("Training Prophet Model...")
if "Date" not in df.columns:
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))

trend_df = df.groupby("Date")["Estimated_Deliveries"].sum().reset_index()
trend_df.rename(columns={"Date": "ds", "Estimated_Deliveries": "y"}, inplace=True)

prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(trend_df)

prophet_model_path = os.path.join(MODEL_DIR, "prophet_model.pkl")
joblib.dump(prophet_model, prophet_model_path)
print(f"Prophet Model saved to {prophet_model_path}")

print("All models trained and saved successfully!")
