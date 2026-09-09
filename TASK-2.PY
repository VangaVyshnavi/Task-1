```python
# ============================================================
# AI & ML INTERNSHIP - TASK 2
# Exploratory Data Analysis (EDA)
# Dataset: Titanic Dataset
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly
import plotly.express as px

# ============================================================
# 2. Load Dataset
# ============================================================

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
df.info()

# ============================================================
# 4. Check Missing Values
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# ============================================================
# 5. Summary Statistics
# ============================================================

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MEDIAN ==========")
print(df.median(numeric_only=True))

print("\n========== STANDARD DEVIATION ==========")
print(df.std(numeric_only=True))

print("\n========== MEAN ==========")
print(df.mean(numeric_only=True))

# ============================================================
# 6. Value Counts for Categorical Variables
# ============================================================

print("\n========== SURVIVAL COUNT ==========")
print(df["Survived"].value_counts())

print("\n========== GENDER COUNT ==========")
print(df["Sex"].value_counts())

print("\n========== CLASS COUNT ==========")
print(df["Pclass"].value_counts())

print("\n========== EMBARKED COUNT ==========")
print(df["Embarked"].value_counts())

# ============================================================
# 7. Histograms for Numeric Features
# ============================================================

numeric_columns = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

for column in numeric_columns:
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# ============================================================
# 8. Boxplots for Numeric Features
# ============================================================

for column in numeric_columns:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        y=column
    )

    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)
    plt.show()

# ============================================================
# 9. Survival Distribution
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# ============================================================
# 10. Survival by Gender
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# ============================================================
# 11. Survival by Passenger Class
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

# ============================================================
# 12. Age Distribution by Survival
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Survived",
    kde=True,
    bins=30
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ============================================================
# 13. Fare Distribution by Survival
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Survived",
    y="Fare"
)

plt.title("Fare Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Fare")
plt.show()

# ============================================================
# 14. Pairplot
# ============================================================

pairplot_columns = [
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

sns.pairplot(
    df[pairplot_columns].dropna(),
    hue="Survived"
)

plt.show()

# ============================================================
# 15. Correlation Matrix
# ============================================================

correlation = df.select_dtypes(
    include=np.number
).corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()

# ============================================================
# 16. Plotly Interactive Scatter Plot
# ============================================================

fig = px.scatter(
    df,
    x="Age",
    y="Fare",
    color="Survived",
    title="Age vs Fare by Survival",
    hover_data=["Sex", "Pclass"]
)

fig.show()

# ============================================================
# 17. Plotly Interactive Bar Chart
# ============================================================

survival_by_class = (
    df.groupby(["Pclass", "Survived"])
      .size()
      .reset_index(name="Count")
)

fig = px.bar(
    survival_by_class,
    x="Pclass",
    y="Count",
    color="Survived",
    barmode="group",
    title="Survival Count by Passenger Class"
)

fig.show()

# ============================================================
# 18. Group Analysis
# ============================================================

print("\n========== SURVIVAL RATE BY GENDER ==========")

gender_survival = df.groupby("Sex")["Survived"].mean()

print(gender_survival)

print("\n========== SURVIVAL RATE BY CLASS ==========")

class_survival = df.groupby("Pclass")["Survived"].mean()

print(class_survival)

print("\n========== AVERAGE FARE BY CLASS ==========")

fare_by_class = df.groupby("Pclass")["Fare"].mean()

print(fare_by_class)

print("\n========== AVERAGE AGE BY CLASS ==========")

age_by_class = df.groupby("Pclass")["Age"].mean()

print(age_by_class)

# ============================================================
# 19. Detect Skewness
# ============================================================

print("\n========== SKEWNESS ==========")

skewness = df.select_dtypes(
    include=np.number
).skew()

print(skewness)

# ============================================================
# 20. Identify Outliers Using IQR
# ============================================================

print("\n========== OUTLIER ANALYSIS ==========")

for column in numeric_columns:

    data = df[column].dropna()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data < lower_bound) |
        (data > upper_bound)
    ]

    print(f"\n{column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers:", len(outliers))

# ============================================================
# 21. Basic Feature-Level Inferences
# ============================================================

print("\n========== BASIC EDA INFERENCES ==========")

print("""
1. The Titanic dataset contains passenger information
   including age, gender, passenger class and fare.

2. Survival is the target variable where:
   0 = Did not survive
   1 = Survived.

3. Survival can be compared across gender and passenger class.

4. Histograms help understand the distribution of numerical
   variables such as Age and Fare.

5. Boxplots help identify possible outliers.

6. The correlation matrix shows relationships between
   numerical features.

7. Skewness values help identify whether numerical features
   have asymmetric distributions.

8. Pairplots help visually understand relationships between
   multiple numerical variables.

9. Grouping the data by gender and passenger class helps
   identify differences in survival rates.
""")

# ============================================================
# 22. Save EDA Summary
# ============================================================

correlation.to_csv("correlation_matrix.csv")

print("\n========== TASK 2 COMPLETED ==========")
print("EDA analysis completed successfully.")
```
