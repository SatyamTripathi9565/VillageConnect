import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load Dataset
df = pd.read_csv("Week9/villageconnect_large.csv")  

# Basic Cleaning
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True)) 
df = df.fillna("Unknown")                   

# Features & Target
X = df.drop(columns=["Satisfaction_Score"])
y = df["Satisfaction_Score"]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Polynomial features for numerical
poly_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', poly_transformer, numerical_cols),            
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  
    ]
)

# Model
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse'
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Model
grid_search.fit(X_train, y_train)

# Best Params
print(" Best parameters found:", grid_search.best_params_)

# Predictions
y_pred = grid_search.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Test MSE: {mse:.4f}")
print(f" R² Score: {r2:.4f}")

# Save the best model for Streamlit
joblib.dump(grid_search.best_estimator_, "village_model_best.pkl")
print(" Model saved as village_model_best.pkl")

# Heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target
print("\n Correlation with Satisfaction_Score:")
print(corr_matrix["Satisfaction_Score"].sort_values(ascending=False))
