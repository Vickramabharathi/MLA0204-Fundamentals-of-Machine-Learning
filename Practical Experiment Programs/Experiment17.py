import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "RAM": np.random.choice([2, 4, 6, 8, 12], n),
    "Storage": np.random.choice([32, 64, 128, 256], n),
    "Battery": np.random.randint(3000, 7000, n),
    "Camera": np.random.choice([12, 16, 32, 48, 64], n)
})

def price_category(row):
    score = 0
    if row["RAM"] >= 8:
        score += 1
    if row["Storage"] >= 128:
        score += 1
    if row["Battery"] >= 5000:
        score += 1
    if row["Camera"] >= 48:
        score += 1

    if score >= 3:
        return "High"
    elif score == 2:
        return "Medium"
    else:
        return "Low"

data["Price"] = data.apply(price_category, axis=1)

X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))