from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

model = GaussianMixture(
    n_components=3,
    random_state=42
)

model.fit(X)

y_pred = model.predict(X)

labels = np.zeros_like(y_pred)

for i in range(3):
    mask = (y_pred == i)
    if np.any(mask):
        labels[mask] = mode(y[mask], keepdims=True).mode[0]

print("Accuracy:", accuracy_score(y, labels))

print("\nPredicted Clusters:")
print(labels)