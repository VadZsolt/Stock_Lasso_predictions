from sklearn.model_selection import train_test_split
from read_data import read_data
from LassoRegression import LassoRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_scaled, y = read_data("../data/data.csv")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

lasso = LassoRegression(alpha=0.5, lr=0.01, max_iter=100000, tol=1e-5)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"RMSE: {rmse}")

results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(results.sort_index())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="red", linestyle="dashed")
plt.title("Actual vs Predicted")
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.legend()
plt.show()