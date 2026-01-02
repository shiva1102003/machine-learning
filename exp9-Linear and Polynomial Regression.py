# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])   # y = x^2 (non-linear)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Performance metrics
lin_mse = mean_squared_error(y, y_lin_pred)
poly_mse = mean_squared_error(y, y_poly_pred)

lin_r2 = r2_score(y, y_lin_pred)
poly_r2 = r2_score(y, y_poly_pred)

# Print results
print("Linear Regression MSE:", lin_mse)
print("Linear Regression R2 Score:", lin_r2)

print("\nPolynomial Regression MSE:", poly_mse)
print("Polynomial Regression R2 Score:", poly_r2)

# Plot results
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_lin_pred, label="Linear Regression", color="red")
plt.plot(X, y_poly_pred, label="Polynomial Regression", color="green")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
