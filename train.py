# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import json
import seaborn as sns

# Load dataset
df = sns.load_dataset("penguins")
print("Initial dataset shape:", df.shape)

# Drop missing values
df = df.dropna()

# Encode target label
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["sex", "island"])

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Save column names and label classes for inference
metadata = {
    "features": list(X.columns),
    "classes": list(le.classes_)
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBClassifier(
    n_estimators=3,
    max_depth=2,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0,
)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nTrain Classification Report:")
print(classification_report(y_train, train_pred, target_names=le.classes_))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Save model and metadata
os.makedirs("app/data", exist_ok=True)
model.save_model("app/data/model.json")

with open("app/data/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Model and metadata saved in 'app/data/'.")
