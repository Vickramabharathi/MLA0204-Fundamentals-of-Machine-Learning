import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "Year": np.random.randint(2010, 2024, n),
    "Mileage": np.random.randint(10000, 200000, n),
    "Engine": np.random.randint(1000, 3000, n),
    "Price": np.random.randint(300000, 1500000, n)
})

X = data[["Year", "Mileage", "Engine"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nActual Price\tPredicted Price")
for actual, predicted in zip(y_test.values[:10], y_pred[:10]):
    print(f"{actual:.0f}\t\t{predicted:.0f}")