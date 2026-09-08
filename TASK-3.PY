```python
# ============================================================
# AI & ML INTERNSHIP - TASK 3
# Linear Regression
# Dataset: House Price Prediction Dataset
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv("house_prices.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ============================================================
# 3. Data Preprocessing
# ============================================================

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

# ============================================================
# 4. Select Features and Target
# ============================================================

# Change these column names if your dataset uses different names.
# Example:
# X = df[["Area"]]
# y = df["Price"]

X = df[["Area"]]
y = df["Price"]

print("\n========== FEATURES ==========")
print(X.head())

print("\n========== TARGET ==========")
print(y.head())

# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAINING DATA ==========")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n========== TEST DATA ==========")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# 6. Create Linear Regression Model
# ============================================================

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# ============================================================
# 7. Make Predictions
# ============================================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(results.head(10))

# ============================================================
# 8. Model Coefficients
# ============================================================

print("\n========== MODEL COEFFICIENTS ==========")

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

print("\nRegression Equation:")
print(
    f"Price = {model.intercept_:.2f} + "
    f"({model.coef_[0]:.2f} × Area)"
)

# ============================================================
# 9. Evaluate Model
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# ============================================================
# 10. Plot Regression Line
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test,
    y_test,
    label="Actual Prices"
)

# Sort values so regression line is displayed correctly
sorted_indices = X_test["Area"].argsort()

X_sorted = X_test.iloc[sorted_indices]

y_sorted_pred = model.predict(X_sorted)

plt.plot(
    X_sorted,
    y_sorted_pred,
    label="Regression Line"
)

plt.xlabel("Area")
plt.ylabel("House Price")

plt.title("Linear Regression - Area vs House Price")

plt.legend()

plt.show()

# ============================================================
# 11. Actual vs Predicted Values
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.show()

# ============================================================
# 12. Residual Analysis
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()

# ============================================================
# 13. Multiple Linear Regression
# ============================================================

# Use additional numerical features if they exist.
possible_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking"
]

available_features = [
    column for column in possible_features
    if column in df.columns
]

if len(available_features) >= 2:

    print("\n========== MULTIPLE LINEAR REGRESSION ==========")

    X_multiple = df[available_features]
    y_multiple = df["Price"]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multiple,
        y_multiple,
        test_size=0.2,
        random_state=42
    )

    multiple_model = LinearRegression()

    multiple_model.fit(
        X_train_m,
        y_train_m
    )

    y_pred_m = multiple_model.predict(X_test_m)

    mae_m = mean_absolute_error(
        y_test_m,
        y_pred_m
    )

    mse_m = mean_squared_error(
        y_test_m,
        y_pred_m
    )

    r2_m = r2_score(
        y_test_m,
        y_pred_m
    )

    print("Features used:", available_features)
    print("MAE:", mae_m)
    print("MSE:", mse_m)
    print("R² Score:", r2_m)

    print("\n========== MULTIPLE REGRESSION COEFFICIENTS ==========")

    for feature, coefficient in zip(
        available_features,
        multiple_model.coef_
    ):
        print(
            f"{feature}: {coefficient:.4f}"
        )

# ============================================================
# 14. Save Predictions
# ============================================================

results.to_csv(
    "linear_regression_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("linear_regression_predictions.csv")

# ============================================================
# 15. Final Interpretation
# ============================================================

print("\n========== MODEL INTERPRETATION ==========")

print(f"""
The Linear Regression model predicts house prices based
on the selected features.

MAE = {mae:.2f}

MSE = {mse:.2f}

R² = {r2:.4f}

The coefficient for Area is:
{model.coef_[0]:.2f}

This means that, according to the fitted model, a one-unit
increase in Area is associated with an estimated change of
{model.coef_[0]:.2f} units in the predicted house price,
while holding the model structure constant.

The R² score indicates how much of the variation in the
target variable is explained by the model.
""")

print("\n========== TASK 3 COMPLETED ==========")
```
