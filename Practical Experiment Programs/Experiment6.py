
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("naive_bayes_dataset.csv")

X = df.drop("PlayTennis", axis=1)
y = df["PlayTennis"]

encoders = {}

for col in X.columns:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col])

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = CategoricalNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))
