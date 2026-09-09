# Task 6: K-Nearest Neighbors (KNN) Classification
# AI & ML Internship

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print("Dataset:")
print(X.head())

print("\nTarget Classes:")
print(iris.target_names)


# ---------------------------------------------------
# 2. Split Dataset into Training and Testing Sets
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# 3. Normalize Features
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------
# 4. Experiment with Different Values of K
# ---------------------------------------------------

k_values = range(1, 21)
accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

    print(f"K = {k:2d} | Accuracy = {accuracy:.4f}")


# ---------------------------------------------------
# 5. Find Best K
# ---------------------------------------------------

best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print("\nBest K:", best_k)
print("Best Accuracy:", best_accuracy)


# ---------------------------------------------------
# 6. Train Final KNN Model
# ---------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)


# ---------------------------------------------------
# 7. Accuracy
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Model Accuracy:", accuracy)


# ---------------------------------------------------
# 8. Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title(f"KNN Confusion Matrix (K={best_k})")
plt.show()


# ---------------------------------------------------
# 9. Plot Accuracy vs K
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(k_values, accuracies, marker="o")

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")

plt.xticks(k_values)
plt.grid(True)

plt.show()


# ---------------------------------------------------
# 10. Decision Boundary Visualization
# ---------------------------------------------------
# Use first two features for 2D visualization.

X_2d = X.iloc[:, :2]

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_2d = StandardScaler()

X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

knn_2d = KNeighborsClassifier(n_neighbors=best_k)

knn_2d.fit(X_train_2d_scaled, y_train_2d)


# Create mesh grid

x_min = X_train_2d_scaled[:, 0].min() - 1
x_max = X_train_2d_scaled[:, 0].max() + 1

y_min = X_train_2d_scaled[:, 1].min() - 1
y_max = X_train_2d_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)


# Predict each point in the grid

Z = knn_2d.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)


# Plot decision boundary

plt.figure(figsize=(9, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

scatter = plt.scatter(
    X_train_2d_scaled[:, 0],
    X_train_2d_scaled[:, 1],
    c=y_train_2d,
    edgecolor="k"
)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.title(
    f"KNN Decision Boundary (K={best_k})"
)

plt.show()


# ---------------------------------------------------
# 11. Final Summary
# ---------------------------------------------------

print("\n========== FINAL RESULTS ==========")
print("Algorithm       : K-Nearest Neighbors")
print("Dataset         : Iris Dataset")
print("Best K          :", best_k)
print("Accuracy        :", round(accuracy, 4))
print("Number of Test Samples :", len(y_test))
print("===================================")