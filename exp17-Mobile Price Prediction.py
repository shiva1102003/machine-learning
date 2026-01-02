# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Create a sample mobile dataset
data = {
    'Battery_Power': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'RAM': [1, 2, 3, 4, 6, 8, 12, 16],
    'Internal_Memory': [8, 16, 32, 64, 128, 256, 256, 512],
    'Camera_MP': [5, 8, 12, 16, 32, 48, 64, 108],
    'Price_Range': [0, 0, 1, 1, 2, 2, 3, 3]  # 0=Low, 1=Medium, 2=High, 3=Very High
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Battery_Power', 'RAM', 'Internal_Memory', 'Camera_MP']]
y = df['Price_Range']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
rf.fit(X_train, y_train)

# Predict price range
y_pred = rf.predict(X_test)

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(cm)

print("\nAccuracy of the Model:")
print(accuracy)
