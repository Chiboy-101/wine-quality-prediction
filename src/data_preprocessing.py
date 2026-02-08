# data_preprocessing.py
# Handles data loading, cleaning, feature engineering, and scaling

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path="data/winequality-red.csv"):
    """
    Load dataset from CSV
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Drop duplicates and handle missing values
    """
    df = df.drop_duplicates()
    # If any missing values exist
    df = df.dropna()
    return df


def feature_engineering(df):
    """
    Add custom features for better model performance
    """
    # Basic feature engineering
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
    df["acid_sugar_ratio"] = df["fixed acidity"] / (df["residual sugar"] + 1e-6)
    df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-6)
    df["sulphate_effect"] = df["sulphates"] * df["alcohol"]

    # Extra safe feature engineering
    df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    df["acidity_balance"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    df["alcohol_volatile_interaction"] = df["alcohol"] * df["volatile acidity"]
    df["log_residual_sugar"] = np.log1p(df["residual sugar"])

    return df


def split_features_target(df, target="quality"):
    """
    Split DataFrame into X and y
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def scale_features(X_train, X_test):
    """
    Standard scale features (mainly for Linear Regression)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
