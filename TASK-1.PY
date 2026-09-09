# ============================================================
# AI & ML INTERNSHIP - TASK 1
# Data Cleaning & Preprocessing
# Dataset: Titanic Dataset
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# ============================================================
# 2. Load Dataset
# ============================================================

# Download Titanic dataset and place it in the same folder
# as this Python file.
df = pd.read_csv("Titanic-Dataset.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# ============================================================
# 3. Basic Information
# ============================================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# ============================================================
# 4. Check Missing Values
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# ============================================================
# 5. Handle Missing Values
# ============================================================

# Age - fill missing values using median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked - fill missing values using mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fare - fill missing values using median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Cabin has many missing values, so drop the column
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

# ============================================================
# 6. Remove Duplicate Rows
# ============================================================

print("\n========== DUPLICATES ==========")
print("Number of duplicate rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after removal:", df.duplicated().sum())

# ============================================================
# 7. Convert Categorical Variables into Numerical
# ============================================================

# Convert Sex using label-style mapping
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True,
    dtype=int
)

print("\n========== DATA AFTER ENCODING ==========")
print(df.head())

# ============================================================
# 8. Drop Unnecessary Columns
# ============================================================

# These columns are identifiers/text and are not required
# for the preprocessing task.
columns_to_drop = ["Name", "Ticket", "PassengerId"]

for column in columns_to_drop:
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)

print("\n========== COLUMNS AFTER CLEANING ==========")
print(df.columns)

# ============================================================
# 9. Visualize Outliers Using Boxplots
# ============================================================

numerical_columns = ["Age", "SibSp", "Parch", "Fare"]

plt.figure(figsize=(12, 8))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column])
    plt.title(f"Boxplot of {column}")

plt.tight_layout()
plt.show()

# ============================================================
# 10. Remove Outliers Using IQR
# ============================================================

def remove_outliers_iqr(data, columns):
    cleaned_data = data.copy()

    for column in columns:
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        cleaned_data = cleaned_data[
            (cleaned_data[column] >= lower_bound) &
            (cleaned_data[column] <= upper_bound)
        ]

    return cleaned_data


print("\n========== SHAPE BEFORE OUTLIER REMOVAL ==========")
print(df.shape)

df = remove_outliers_iqr(df, numerical_columns)

print("\n========== SHAPE AFTER OUTLIER REMOVAL ==========")
print(df.shape)

# ============================================================
# 11. Standardize Numerical Features
# ============================================================

scaler = StandardScaler()

df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)

print("\n========== DATA AFTER STANDARDIZATION ==========")
print(df.head())

# ============================================================
# 12. Check Final Dataset
# ============================================================

print("\n========== FINAL DATASET INFORMATION ==========")
print(df.info())

print("\n========== FINAL MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== FINAL STATISTICS ==========")
print(df.describe())

# ============================================================
# 13. Save Cleaned Dataset
# ============================================================

df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved as: titanic_cleaned.csv")

# ============================================================
# 14. Final Boxplots After Cleaning
# ============================================================

plt.figure(figsize=(12, 8))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column])
    plt.title(f"After Preprocessing - {column}")

plt.tight_layout()
plt.show()

print("\n========== TASK 1 COMPLETED ==========")