
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("knn_dataset.csv")

X = df[["Feature1", "Feature2"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:", y_test.values)
print("Predicted Values:", y_pred)
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

sample = pd.DataFrame([[1.0, 1.1]], columns=["Feature1", "Feature2"])
sample_scaled = scaler.transform(sample)
print("New Sample Prediction:", model.predict(sample_scaled)[0])
