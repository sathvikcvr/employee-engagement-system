
import pandas as pd
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv(
    "Employee Engagement System/Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ],
    inplace=True
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

# 2. Identify feature types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 3. Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numeric_features
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )
    )
])

# 6. Hyperparameter tuning
param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring={
        "f1": "f1",
        "recall": "recall",
        "roc_auc": "roc_auc"
    },
    refit="f1",
    cv=cv,
    n_jobs=-1
)

print("Starting hyperparameter tuning...")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest cross-validation F1:")
print(grid.best_score_)

# 7. Get out-of-fold probabilities
# Training data only; test data remains untouched.
oof_probabilities = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# 8. Find threshold with best F1
thresholds = [i / 100 for i in range(10, 91, 5)]

threshold_results = []

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_threshold = threshold_df.loc[
    threshold_df["F1"].idxmax(),
    "Threshold"
]

print("\nThreshold comparison:")
print(threshold_df.round(3).to_string(index=False))

print("\nSelected threshold:", best_threshold)

# 9. Final evaluation on untouched test data
best_model.fit(X_train, y_train)

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("\nAccuracy:",
      accuracy_score(y_test, test_predictions))

print("\nPrecision:",
      precision_score(y_test, test_predictions))

print("\nRecall:",
      recall_score(y_test, test_predictions))

print("\nF1 Score:",
      f1_score(y_test, test_predictions))

print("\nROC-AUC:",
      roc_auc_score(y_test, test_probabilities))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Stayed", "Left"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# 10. Save model and threshold
os.makedirs("models", exist_ok=True)

with open("models/best_attrition_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/best_threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("\nOptimized model saved successfully!")
