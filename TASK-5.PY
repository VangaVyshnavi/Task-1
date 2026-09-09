```python
# ============================================================
# TASK 4: CLASSIFICATION WITH LOGISTIC REGRESSION
# AI & ML INTERNSHIP
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)


# -----------------------------
# 2. Load Dataset
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset loaded successfully!")
print("\nDataset Shape:", X.shape)

print("\nFirst 5 rows:")
print(X.head())

print("\nTarget Classes:")
print(data.target_names)

print("\nClass Distribution:")
print(y.value_counts())


# -----------------------------
# 3. Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(X.isnull().sum().sum())


# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -----------------------------
# 5. Standardize Features
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 6. Train Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")


# -----------------------------
# 7. Make Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)

# Probability of positive class
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# -----------------------------
# 8. Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))


# -----------------------------
# 9. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# -----------------------------
# 10. Precision and Recall
# -----------------------------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))


# -----------------------------
# 11. Classification Report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# -----------------------------
# 12. ROC-AUC Score
# -----------------------------
roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", round(roc_auc, 4))


# -----------------------------
# 13. Plot ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# 14. Sigmoid Function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-10, 10, 200)
probabilities = sigmoid(z)

plt.figure(figsize=(8, 6))

plt.plot(z, probabilities)

plt.axhline(
    y=0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.xlabel("Input (z)")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# 15. Threshold Tuning
# -----------------------------
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\nThreshold Tuning:")
print("-" * 50)

for threshold in thresholds_to_test:

    y_threshold_pred = (y_prob >= threshold).astype(int)

    threshold_precision = precision_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {threshold_precision:.4f} | "
        f"Recall: {threshold_recall:.4f}"
    )


# -----------------------------
# 16. Find Best Threshold
# -----------------------------
best_threshold = 0.5
best_recall = 0

for threshold in np.arange(0.1, 1.0, 0.01):

    predictions = (y_prob >= threshold).astype(int)

    current_recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    if current_recall > best_recall:
        best_recall = current_recall
        best_threshold = threshold

print("\nBest Threshold for Maximum Recall:",
      round(best_threshold, 2))

print("Maximum Recall:",
      round(best_recall, 4))


# -----------------------------
# 17. Coefficients
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop Important Features:")
print(coefficients.head(10))


# -----------------------------
# 18. Save Predictions
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_prob
})

results.to_csv(
    "logistic_regression_predictions.csv",
    index=False
)

print(
    "\nPredictions saved as "
    "'logistic_regression_predictions.csv'"
)


# -----------------------------
# 19. Final Results
# -----------------------------
print("\n" + "=" * 50)
print("FINAL MODEL RESULTS")
print("=" * 50)

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("ROC-AUC   :", round(roc_auc, 4))
print("=" * 50)
```
