# model_tuning.py
# Hyperparameter tuning for Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle

from data_preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    split_features_target,
)
from utils import evaluate_model, plot_feature_importance


# Load and prepare data
df = load_data()
df = clean_data(df)
df = feature_engineering(df)

X, y = split_features_target(df)


# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Hyperparameter tuning for Random Forest
param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

random_search.fit(X_train, y_train)

print("\nBest Parameters found:", random_search.best_params_)
print("Best Cross-Validation R²:", random_search.best_score_)

best_rf = random_search.best_estimator_


# Evaluate tuned model
y_pred_tuned = best_rf.predict(X_test)
metrics = evaluate_model(y_test, y_pred_tuned)

print("\nTuned Random Forest Test Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# Feature importance plot
plot_feature_importance(
    best_rf, X.columns, top_n=10, title="Top 10 Feature Importances"
)


# Save model
with open("models/best_wine_quality_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)
