import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("ann_backpropagation_dataset.csv")

X = df[["Feature1", "Feature2"]].values
y = df[["Target"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.random.seed(42)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 5000

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

for epoch in range(epochs):
    hidden = sigmoid(X_train @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    error = output - y_train

    dW2 = hidden.T @ error / len(X_train)
    db2 = np.mean(error, axis=0, keepdims=True)

    dhidden = (error @ W2.T) * hidden * (1 - hidden)
    dW1 = X_train.T @ dhidden / len(X_train)
    db1 = np.mean(dhidden, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

hidden_test = sigmoid(X_test @ W1 + b1)
output_test = sigmoid(hidden_test @ W2 + b2)

y_pred = (output_test >= 0.5).astype(int)

print("Actual Values:   ", y_test.flatten())
print("Predicted Values:", y_pred.flatten())
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))