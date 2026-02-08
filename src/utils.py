# utils.py
# Utility functions for Wine Quality Prediction Project

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Metrics Calculation Function
def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model performance.

    Parameters:
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values

    Returns:
    dict : Dictionary containing RMSE, R², MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {"RMSE": rmse, "R2": r2, "MAE": mae}
    return metrics


# Plot Residuals
def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals of regression predictions.

    Parameters:
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str
        Plot title
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.show()


# Feature Importance Plot
def plot_feature_importance(
    model, feature_names, top_n=10, title="Top Feature Importances"
):
    """
    Plot feature importances for tree-based models.

    Parameters:
    model : trained tree-based model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to plot
    title : str
        Plot title
    """
    importances = model.feature_importances_
    feat_imp = {name: imp for name, imp in zip(feature_names, importances)}
    feat_imp_sorted = dict(
        sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=list(feat_imp_sorted.values()),
        y=list(feat_imp_sorted.keys()),
        palette="viridis",
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


# Correlation Heatmap
def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """
    Plot correlation heatmap for a DataFrame.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing features
    title : str
        Plot title
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title(title)
    plt.show()
