import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "TV": np.random.randint(10000, 100000, n),
    "Radio": np.random.randint(5000, 50000, n),
    "Newspaper": np.random.randint(2000, 30000, n)
})

data["Sales"] = (
    5
    + 0.0002 * data["TV"]
    + 0.0003 * data["Radio"]
    + 0.0001 * data["Newspaper"]
    + np.random.normal(0, 2, n)
)

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nActual Sales\tPredicted Sales")
for actual, predicted in zip(y_test.values[:10], y_pred[:10]):
    print(f"{actual:.2f}\t\t{predicted:.2f}")