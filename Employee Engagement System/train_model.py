
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove unnecessary columns
df.drop(
    columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
    inplace=True
)

# Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE ONLY to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- SMOTE MODEL RESULTS ---")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Save model
os.makedirs("models", exist_ok=True)

with open("models/attrition_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nSMOTE model saved successfully!")
