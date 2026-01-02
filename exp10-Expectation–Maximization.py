# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Create sample dataset
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3]
])

# Initialize Gaussian Mixture Model with 2 clusters
gmm = GaussianMixture(n_components=2, random_state=42)

# Fit the model (EM algorithm runs internally)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Print cluster labels
print("Cluster Labels:")
print(labels)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("EM Algorithm using Gaussian Mixture Model")
plt.show()
