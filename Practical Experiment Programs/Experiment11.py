import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "Age": np.random.randint(21, 60, n),
    "Income": np.random.randint(30000, 100000, n),
    "Loan_Amount": np.random.randint(50000, 300000, n),
    "Credit_History": np.random.choice(["Good", "Poor"], n)
})

def credit_score(row):
    if row["Income"] > 70000 and row["Credit_History"] == "Good":
        return "Good"
    elif row["Income"] > 50000:
        return "Standard"
    else:
        return "Poor"

data["Credit_Score"] = data.apply(credit_score, axis=1)

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))