import pandas as pd
import math
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("play_tennis.csv")

def entropy(target):
    values = Counter(target)
    total = len(target)
    ent = 0

    for count in values.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent

def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy

target = "Target"

print("Dataset:")
print(data)

print("\nInformation Gain:")

for column in data.columns:
    if column != target:
        gain = information_gain(data, column, target)
        print(column, "=", round(gain, 4))

X = data.drop(target, axis=1)
y = data[target]

X = pd.get_dummies(X)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X, y)

new_sample = pd.DataFrame({
    "Outlook": ["sunny"],
    "Temperature": ["cool"],
    "Humidity": ["high"],
    "Wind": ["strong"]
})

new_sample = pd.get_dummies(new_sample)
new_sample = new_sample.reindex(columns=X.columns, fill_value=False)

prediction = model.predict(new_sample)

print("\nNew Sample:")
print(new_sample)

print("\nPredicted Class:", prediction[0])

accuracy = accuracy_score(y, model.predict(X))

print("\nAccuracy:", round(accuracy * 100, 2), "%")

plt.figure(figsize=(12, 7))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("ID3 Decision Tree")
plt.show()
