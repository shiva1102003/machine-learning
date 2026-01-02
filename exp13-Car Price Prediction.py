# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a sample car dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Mileage': [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000],
    'Engine_Size': [1.2, 1.4, 1.6, 1.6, 1.8, 2.0, 2.0, 2.2],
    'Price': [500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Year', 'Mileage', 'Engine_Size']]
y = df['Price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
