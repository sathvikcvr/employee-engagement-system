
import pandas as pd

# Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Dataset overview
print("\n--- DATASET INFORMATION ---")
print(df.info())

# Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check attrition distribution
print("\n--- ATTRITION DISTRIBUTION ---")
print(df["Attrition"].value_counts())

# Display all column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())
