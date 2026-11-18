
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
with open("models/attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("models/model_columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Extract feature importance
importance = pd.DataFrame({
    "Feature": columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 ATTRITION FACTORS")
print(importance.head(15).to_string(index=False))

# Plot top 15 features
plt.figure(figsize=(10, 6))
plt.barh(
    importance.head(15)["Feature"],
    importance.head(15)["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 15 Employee Attrition Factors")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.show()
