# ============================================
# TASK 1: DATA CLEANING & PREPROCESSING
# AI & ML INTERNSHIP
# ============================================

# Step 1: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================
# Step 2: Load the dataset
# ============================================

# Change the file name if your dataset has a different name
df = pd.read_csv("titanic.csv")

print("Dataset loaded successfully!")


# ============================================
# Step 3: Explore the dataset
# ============================================

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display last 5 rows
print("\nLast 5 rows:")
print(df.tail())

# Number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Basic information
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary:")
print(df.describe())


# ============================================
# Step 4: Check missing values
# ============================================

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nMissing value percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)


# ============================================
# Step 5: Handle missing values
# ============================================

# Fill Age missing values using median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked missing values using mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Fare missing values using median
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Drop Cabin because it contains many missing values
if "Cabin" in df.columns:
    df = df.drop("Cabin", axis=1)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# Step 6: Convert categorical data
# ============================================

# One-hot encoding for categorical columns
categorical_columns = []

for column in ["Sex", "Embarked"]:
    if column in df.columns:
        categorical_columns.append(column)

if len(categorical_columns) > 0:
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

print("\nDataset after categorical encoding:")
print(df.head())


# ============================================
# Step 7: Remove unnecessary columns
# ============================================

# These columns are identifiers/text and are not required
columns_to_drop = []

for column in ["Name", "Ticket", "PassengerId"]:
    if column in df.columns:
        columns_to_drop.append(column)

if len(columns_to_drop) > 0:
    df = df.drop(columns=columns_to_drop)

print("\nColumns after removing unnecessary data:")
print(df.columns)


# ============================================
# Step 8: Detect outliers using boxplots
# ============================================

# Select numerical columns
numerical_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical columns:")
print(list(numerical_columns))

# Create boxplots
for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title("Boxplot of " + column)
    plt.xlabel(column)

    plt.show()


# ============================================
# Step 9: Remove outliers using IQR
# ============================================

# We will use IQR method for numerical columns

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower_limit) &
        (df[column] <= upper_limit)
    ]

print("\nShape after removing outliers:")
print(df.shape)


# ============================================
# Step 10: Feature scaling
# ============================================

# Select numerical features
numerical_features = df.select_dtypes(
    include=np.number
).columns

# Standardization
scaler = StandardScaler()

df[numerical_features] = scaler.fit_transform(
    df[numerical_features]
)

print("\nDataset after standardization:")
print(df.head())


# ============================================
# Step 11: Final data check
# ============================================

print("\nFinal dataset information:")
df.info()

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nFinal dataset shape:")
print(df.shape)


# ============================================
# Step 12: Save cleaned dataset
# ============================================

df.to_csv(
    "titanic_cleaned_preprocessed.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\nTASK 1 COMPLETED!")