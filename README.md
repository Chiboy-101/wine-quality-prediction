# 🍷 Red Wine Quality Prediction

A machine learning project that predicts the **quality of red wine** based on its physicochemical properties using **Random Forest Regression** and other models. The project includes **feature engineering, model training, hyperparameter tuning, evaluation, and visualization**.

---

## 📌 Project Overview

The goal of this project is to predict the quality of red wines (score between 0–10) based on features like acidity, alcohol content, residual sugar, and sulphates. This can help wine producers understand which chemical properties contribute most to wine quality.

**Key steps performed in this project:**
1. Data loading and cleaning  
2. Feature engineering for improved model performance  
3. Exploratory Data Analysis (EDA) and correlation analysis  
4. Model training: Linear Regression, Decision Tree, Random Forest, Gradient Boosting  
5. Hyperparameter tuning using `RandomizedSearchCV`  
6. Model evaluation: RMSE, R², MAE  
7. Visualization: Residuals, Feature Importance, Correlation Heatmap  
8. Saving trained model and scaler for deployment  


---

## 📊 Dataset

**Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Columns:**
- `fixed acidity`  
- `volatile acidity`  
- `citric acid`  
- `residual sugar`  
- `chlorides`  
- `free sulfur dioxide`  
- `total sulfur dioxide`  
- `density`  
- `pH`  
- `sulphates`  
- `alcohol`  
- `quality` (target variable)

---

## ⚡ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
jupyter notebook notebooks/wine_quality_analysis.ipynb # for EDA
python src/model_training.py # Train and evaluate models
python src/model_tuning.py # Hyper parameter tuning

