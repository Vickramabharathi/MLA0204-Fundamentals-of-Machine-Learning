import pandas as pd
import numpy as np
from google.colab import files

# Upload CSV file
uploaded = files.upload()

# Read dataset
data = pd.read_csv("training_data.csv")

print("Training Dataset:\n")
print(data)

# Convert dataset into numpy array
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

# Candidate Elimination Algorithm
def candidate_elimination(concepts, target):

    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    print("\nInitial Specific Hypothesis:")
    print(specific_h)

    print("\nInitial General Hypothesis:")
    print(general_h)

    for i, h in enumerate(concepts):

        if target[i].lower() == "yes":

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"

        if target[i].lower() == "no":

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

    # Remove overly general hypotheses
    final_general = []

    for row in general_h:
        if row != ["?" for _ in range(len(specific_h))]:
            final_general.append(row)

    return specific_h, final_general


S, G = candidate_elimination(concepts, target)

print("\nFinal Specific Hypothesis:")
print(S)

print("\nFinal General Hypothesis:")
for g in G:
    print(g)