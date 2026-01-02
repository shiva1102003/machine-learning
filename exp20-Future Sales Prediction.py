# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample sales dataset
data = {
    'Advertising_Spend': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'Previous_Sales': [20000, 24000, 28000, 32000, 36000, 40000, 44000, 48000],
    'Sales': [22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Advertising_Spend', 'Previous_Sales']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict future sales
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted Sales:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
