# FIND-S Algorithm Implementation
def find_s_algorithm(training_data):
    # Step 1: Initialize the most specific hypothesis
    num_attributes = len(training_data[0]) - 1
    hypothesis = ['Ø'] * num_attributes
    print("Initial Hypothesis:")
    print(hypothesis)
    print("-" * 50)
    # Step 2: Iterate through training examples
    for index, example in enumerate(training_data):
        print(f"Training Example {index + 1}: {example}")
        # Consider only positive examples
        if example[-1] == 'Yes':
            for i in range(num_attributes):
                if hypothesis[i] == 'Ø':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
            print("Updated Hypothesis:", hypothesis)
        else:
            print("Negative example - Ignored")
        print("-" * 50)
    return hypothesis
# Training dataset
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]
# Run FIND-S Algorithm
final_hypothesis = find_s_algorithm(training_data)
print("\nFinal Most Specific Hypothesis:")
print(final_hypothesis)
