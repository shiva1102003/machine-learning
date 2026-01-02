import pandas as pd

# Load CSV file
data = pd.read_csv("training_data.csv")

# Separate attributes and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize S and G
S = list(X[0])
G = [['?' for _ in range(len(S))]]

# Candidate Elimination Algorithm
for i in range(len(X)):
    if y[i] == 'Yes':   # Positive example
        for j in range(len(S)):
            if S[j] != X[i][j]:
                S[j] = '?'
                G[0][j] = '?'
                
    else:               # Negative example
        for j in range(len(S)):
            if S[j] != X[i][j]:
                G[0][j] = S[j]
            else:
                G[0][j] = '?'

print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypothesis (G):")
print(G)

