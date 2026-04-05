"""
Crop yield prediction using Random Forest and XGBoost.
Dataset: Kaggle Crop Yield dataset (temperature, rainfall, region, year)
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


FEATURES = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[FEATURES + [TARGET]].dropna()

    le = LabelEncoder()
    df["Area"] = le.fit_transform(df["Area"])
    df["Item"] = le.fit_transform(df["Item"])

    return df


def train(csv_path: str, model_type: str = "xgboost", save_path: str = "results/yield_model.pkl"):
    df = load_data(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "xgboost":
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f} | R²: {r2:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

    _plot_feature_importance(model, X_train.columns)
    return model


def _plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 4))
        plt.barh(feature_names, model.feature_importances_)
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        plt.close()


if __name__ == "__main__":
    train(csv_path="data/raw/yield_df.csv", model_type="xgboost")
