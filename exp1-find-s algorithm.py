import pandas as pd

# Training data
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'PlayTennis'])

# Separate attributes and target
attributes = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Initialize hypothesis with first positive example
hypothesis = None

for i in range(len(data)):
    if target[i] == 'Yes':
        if hypothesis is None:
            hypothesis = list(attributes.iloc[i])
        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes.iloc[i, j]:
                    hypothesis[j] = '?'

print("Most Specific Hypothesis:")
print(hypothesis)

