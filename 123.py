
# ============================================================
# CRISP-DM PROJECT: 50 STARTUPS
# Multiple Linear Regression using Scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    KFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

RANDOM_STATE = 42
DATA_PATH = "50_Startups.csv"


# ============================================================
# STAGE 1: BUSINESS UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1: BUSINESS UNDERSTANDING")
print("=" * 60)

# 1.1 Determine Business Objectives
business_objective = (
    "Predict startup profit using company expenditures "
    "and geographic location."
)

# 1.2 Assess Situation
# Dataset: 50 startup companies
# Limitations: small sample size and historical data

# 1.3 Determine Data Mining Goals
# Type: Supervised Learning
# Problem: Multiple Linear Regression
# Target variable: Profit

# 1.4 Produce Project Plan
# 1. Load and explore dataset
# 2. Prepare data
# 3. Train regression model
# 4. Evaluate model
# 5. Save and use model

print("Business Objective:", business_objective)
print("Machine Learning Type: Supervised Learning")
print("Algorithm: Multiple Linear Regression")
print("Target Variable: Profit")


# ============================================================
# STAGE 2: DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2: DATA UNDERSTANDING")
print("=" * 60)

# 2.1 Collect Initial Data

df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows:")
print(df.head())

# 2.2 Describe Data

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nColumn Names:")
print(df.columns.tolist())

# 2.3 Explore Data

numeric_df = df.select_dtypes(include=np.number)

# Correlation Matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Distribution of Profit
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Profit"],
    bins=10,
    kde=True
)

plt.title("Distribution of Startup Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# R&D Spend vs Profit
plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="R&D Spend",
    y="Profit"
)

plt.title("R&D Spend vs Profit")
plt.tight_layout()
plt.show()

# Marketing Spend vs Profit
plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="Marketing Spend",
    y="Profit"
)

plt.title("Marketing Spend vs Profit")
plt.tight_layout()
plt.show()

# 2.4 Verify Data Quality

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nInvalid Negative Expenses:")
expense_columns = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

for col in expense_columns:
    print(col, (df[col] < 0).sum())


# ============================================================
# STAGE 3: DATA PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("STAGE 3: DATA PREPARATION")
print("=" * 60)

# 3.1 Select Data

required_columns = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State",
    "Profit"
]

df = df[required_columns].copy()

# 3.2 Clean Data

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna(subset=required_columns)

# Remove rows with invalid negative expenditures
for col in expense_columns:
    df = df[df[col] >= 0]

# Convert State to a consistent format
df["State"] = df["State"].str.strip()

print("\nShape after cleaning:")
print(df.shape)

# 3.3 Construct Data

# Independent variables X
X = df.drop(columns=["Profit"])

# Dependent variable y
y = df["Profit"]

numerical_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

categorical_features = ["State"]

# State is a categorical feature
# OneHotEncoder converts it into numerical columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# 3.4 Integrate Data
# Not required because data comes from one CSV file.

# 3.5 Format Data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nInput Features:")
print(X.columns.tolist())


# ============================================================
# STAGE 4: MODELING
# ============================================================

print("\n" + "=" * 60)
print("STAGE 4: MODELING")
print("=" * 60)

# 4.1 Select Modeling Technique
# Multiple Linear Regression

regression_model = LinearRegression()

# Build a pipeline to avoid preprocessing leakage
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regression_model)
    ]
)

# 4.2 Generate Test Design

# Use 5-fold cross-validation on training data only
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error"
    }
)

cv_r2 = cv_results["test_r2"]
cv_mae = -cv_results["test_neg_mae"]
cv_rmse = np.sqrt(-cv_results["test_neg_mse"])

print("\nCross-Validation Results:")
print(f"Mean CV R2: {cv_r2.mean():.4f}")
print(f"Mean CV MAE: {cv_mae.mean():.2f}")
print(f"Mean CV RMSE: {cv_rmse.mean():.2f}")

# 4.3 Build Model

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 4.4 Assess Model

# Get feature names after encoding
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get model coefficients
coefficients = (
    model.named_steps["regressor"].coef_
)

intercept = (
    model.named_steps["regressor"].intercept_
)

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print("\nRegression Coefficients:")
print(coef_df)

print("\nIntercept:")
print(intercept)

# Training predictions
y_train_pred = model.predict(X_train)

print("\nTraining R2:")
print(r2_score(y_train, y_train_pred))


# ============================================================
# STAGE 5: EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STAGE 5: EVALUATION")
print("=" * 60)

# 5.1 Evaluate Results

# Test predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nTest Set Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# Compare against a simple baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

print(f"\nBaseline RMSE: {baseline_rmse:.2f}")
print(f"Linear Regression RMSE: {rmse:.2f}")

if rmse < baseline_rmse:
    print("Model outperforms the baseline.")
else:
    print("Model does not outperform the baseline.")

# Actual vs Predicted DataFrame
results = pd.DataFrame({
    "Actual Profit": y_test.to_numpy(),
    "Predicted Profit": y_pred
})

results["Residual"] = (
    results["Actual Profit"] -
    results["Predicted Profit"]
)

print("\nPrediction Comparison:")
print(results)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    label="Predictions"
)

lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.legend()
plt.tight_layout()
plt.show()

# Residual Analysis
residuals = y_test.to_numpy() - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)
plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Profit")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# Residual Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    residuals,
    bins=8,
    kde=True
)

plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()

# 5.2 Review Process

print("\nProcess Review:")
print("Train/Test Split: 80/20")
print("Cross-Validation: 5-Fold")
print("Categorical Encoding: One-Hot")
print("Model: Multiple Linear Regression")

# 5.3 Determine Next Steps

print("\nEvaluation Decision:")
print("Compare errors against business requirements.")
print("Review cross-validation stability.")
print("Consider collecting more data before production use.")


# ============================================================
# STAGE 6: DEPLOYMENT
# ============================================================

print("\n" + "=" * 60)
print("STAGE 6: DEPLOYMENT")
print("=" * 60)

# 6.1 Plan Deployment

# Save model pipeline
model_path = Path("startup_profit_model.pkl")

joblib.dump(model, model_path)

print("\nModel Saved:")
print(model_path)

# Load model (only load files from trusted sources)
loaded_model = joblib.load(model_path)

# Predict a new startup's profit
new_startup = pd.DataFrame({
    "R&D Spend": [120000],
    "Administration": [130000],
    "Marketing Spend": [250000],
    "State": ["California"]
})

predicted_profit = loaded_model.predict(new_startup)

print("\nNew Startup Data:")
print(new_startup)

print("\nPredicted Profit:")
print(f"${predicted_profit[0]:,.2f}")

# 6.2 Plan Monitoring and Maintenance

monitoring_plan = {
    "Metric": ["MAE", "RMSE", "R2"],
    "Purpose": [
        "Monitor average prediction error",
        "Monitor large prediction errors",
        "Monitor predictive performance"
    ]
}

print("\nMonitoring Plan:")
print(pd.DataFrame(monitoring_plan))

# 6.3 Produce Final Report

report = {
    "Project": "50 Startups Profit Prediction",
    "Algorithm": "Multiple Linear Regression",
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test),
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Baseline RMSE": baseline_rmse,
    "CV Mean RMSE": cv_rmse.mean()
}

report_df = pd.DataFrame(
    list(report.items()),
    columns=["Item", "Value"]
)

report_df.to_csv(
    "crisp_dm_report.csv",
    index=False
)

results.to_csv(
    "prediction_results.csv",
    index=False
)

coef_df.to_csv(
    "regression_coefficients.csv",
    index=False
)

print("\nFinal Report:")
print(report_df)

# 6.4 Review Project

print("\nProject Review:")
print("1. Business objective defined.")
print("2. Dataset explored and checked.")
print("3. Data cleaned and encoded.")
print("4. Regression model trained.")
print("5. Predictions evaluated.")
print("6. Model and reports saved.")

print("\nCRISP-DM PROJECT COMPLETED")
