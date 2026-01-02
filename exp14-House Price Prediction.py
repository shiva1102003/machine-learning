# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a sample house dataset
data = {
    'Area_sqft': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 4, 5],
    'Bathrooms': [1, 2, 2, 2, 3, 3, 3, 4],
    'House_Age': [20, 15, 12, 10, 8, 5, 3, 1],
    'Price': [3000000, 4000000, 5000000, 6000000,
              7000000, 8000000, 9000000, 10000000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Area_sqft', 'Bedrooms', 'Bathrooms', 'House_Age']]
y = df['Price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted House Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
