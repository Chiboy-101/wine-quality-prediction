# model_training.py
# Train and evaluate models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle

from data_preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    split_features_target,
    scale_features,
)
from utils import (
    evaluate_model,
    plot_residuals,
    plot_feature_importance,
    plot_correlation_heatmap,
)


# Load and prepare data
df = load_data()
df = clean_data(df)
df = feature_engineering(df)

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features for Linear Regression
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
}


# Train and evaluate
results = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    results[name] = metrics
    print(
        f"{name}: RMSE -> {metrics['RMSE']:.4f}, R² -> {metrics['R2']:.4f}, MAE -> {metrics['MAE']:.4f}"
    )

    # Plot residuals for tree-based models
    if name != "Linear Regression":
        plot_residuals(y_test, y_pred, title=f"{name} Residual Plot")


# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
