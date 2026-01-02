# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -------------------------------
# Step 1: Create Bank Loan Dataset
# -------------------------------
data = {
    'Age': [22, 25, 30, 35, 40, 45, 50, 55, 60, 28, 33, 48],
    'Income': [20000, 30000, 40000, 50000, 60000, 70000,
               80000, 90000, 100000, 35000, 45000, 75000],
    'Credit_History': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    'Loan_Status': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Features and Target
# -------------------------------
X = df[['Age', 'Income', 'Credit_History']]
y = df['Loan_Status']

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------
# Step 4: Train Naive Bayes Model
# -------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluation
# -------------------------------
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Step 7: Display Results
# -------------------------------
print("Confusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    zero_division=0   
))
