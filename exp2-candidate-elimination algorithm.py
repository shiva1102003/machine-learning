import csv

def load_csv(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    return data[1:]  # remove header


def candidate_elimination(training_data):
    num_attributes = len(training_data[0]) - 1

    # Initialize S and G
    S = ['Ø'] * num_attributes
    G = [['?'] * num_attributes]

    print("Initial S:", S)
    print("Initial G:", G)
    print("-" * 60)

    for index, example in enumerate(training_data):
        attributes = example[:-1]
        label = example[-1]

        print(f"Training Example {index + 1}: {example}")

        # POSITIVE EXAMPLE
        if label == 'Yes':
            # Remove hypotheses from G that do not cover example
            G = [g for g in G if all(g[i] == '?' or g[i] == attributes[i]
                                     for i in range(num_attributes))]

            # Generalize S
            for i in range(num_attributes):
                if S[i] == 'Ø':
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = '?'

        # NEGATIVE EXAMPLE
        else:
            new_G = []
            for g in G:
                if all(g[i] == '?' or g[i] == attributes[i]
                       for i in range(num_attributes)):
                    for i in range(num_attributes):
                        if g[i] == '?' and S[i] != attributes[i]:
                            new_hypothesis = g.copy()
                            new_hypothesis[i] = S[i]
                            if new_hypothesis not in new_G:
                                new_G.append(new_hypothesis)
                else:
                    new_G.append(g)

            G = new_G

        print("S:", S)
        print("G:", G)
        print("-" * 60)

    return S, G


# ================== MAIN PROGRAM ==================

filename = "training_data.csv"
training_data = load_csv(filename)

S, G = candidate_elimination(training_data)

print("\nFinal Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")
for hypothesis in G:
    print(hypothesis)
