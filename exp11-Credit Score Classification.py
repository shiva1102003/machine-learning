# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Create a sample credit dataset
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 55],
    'Income': [30000, 80000, 50000, 90000, 20000, 70000, 100000, 85000, 45000, 95000],
    'Loan_Amount': [150000, 100000, 200000, 120000, 250000, 130000, 90000, 110000, 220000, 100000],
    'Credit_History': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    'Credit_Score': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]  # 1 = Good, 0 = Bad
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Age', 'Income', 'Loan_Amount', 'Credit_History']]
y = df['Credit_Score']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree model
dt = DecisionTreeClassifier(criterion='entropy')

# Train the model
dt.fit(X_train, y_train)

# Predict on test data
y_pred = dt.predict(X_test)

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(cm)

print("\nAccuracy of the Model:")
print(accuracy)
