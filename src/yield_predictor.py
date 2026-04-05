"""
Crop yield prediction using XGBoost.
Dataset: Agri_yield_prediction.csv
Features: weather (from UI sliders) + soil/crop data (from farmer CSV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

# Features used for prediction
# Weather features come from the UI sliders; soil/crop features from the farmer CSV
NUMERIC_FEATURES = [
    "Temperature", "Humidity", "Rainfall",
    "pH", "N", "P", "K",
    "Irrigation_Frequency",
]
CATEGORICAL_FEATURES = ["Crop_Type", "Soil_Type", "Fertilizer_Type", "Pesticide_Usage"]
ALL_FEATURES = NUMERIC_FEATURES + [f"{c}_enc" for c in CATEGORICAL_FEATURES]
TARGET = "Yield"


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].dropna()

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def train(csv_path: str = "data/raw/Agri_yield_prediction.csv",
          model_type: str = "xgboost",
          save_path: str = "results/yield_model.pkl"):
    df, encoders = load_data(csv_path)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "xgboost":
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.4f} t/ha | R²: {r2:.4f}")

    bundle = {"model": model, "encoders": encoders, "features": ALL_FEATURES}
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {save_path}")

    _plot_feature_importance(model, ALL_FEATURES)
    return bundle


def _plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, model.feature_importances_)
        plt.xlabel("Importance")
        plt.title("Feature Importance — Yield Model")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        plt.close()


if __name__ == "__main__":
    train(csv_path="data/raw/Agri_yield_prediction.csv", model_type="xgboost")
