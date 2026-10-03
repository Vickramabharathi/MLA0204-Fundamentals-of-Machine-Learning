import pandas as pd
from google.colab import files

# Upload CSV
uploaded = files.upload()

# Read CSV
data = pd.read_csv("training_data1.csv")

print("Training Dataset:\n")
print(data)

# Initialize hypothesis
hypothesis = ['0'] * (len(data.columns) - 1)

# Find-S Algorithm
for _, row in data.iterrows():
    if row["EnjoySport"] == "Yes":
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row.iloc[i]
            elif hypothesis[i] != row.iloc[i]:
                hypothesis[i] = '?'

print("\nMost Specific Hypothesis:")
print(hypothesis)