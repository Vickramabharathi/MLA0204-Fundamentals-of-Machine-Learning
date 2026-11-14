import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "Age": np.random.randint(21, 60, n),
    "Income": np.random.randint(20000, 100000, n),
    "LoanAmount": np.random.randint(50000, 500000, n),
    "CreditHistory": np.random.choice(["Good", "Bad"], n)
})

def loan_status(row):
    if row["Income"] > 60000 and row["CreditHistory"] == "Good":
        return "Approved"
    else:
        return "Rejected"

data["LoanStatus"] = data.apply(loan_status, axis=1)

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

X = data.drop("LoanStatus", axis=1)
y = data["LoanStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))