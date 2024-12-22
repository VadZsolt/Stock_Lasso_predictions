import numpy as np

class LassoRegression:
    def __init__(self, alpha, lr, max_iter, tol):
        self.alpha = alpha  # Regularization strength
        self.lr = lr        # Learning rate
        self.max_iter = max_iter  # Maximum iterations
        self.tol = tol      # Convergence tolerance
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iter):
            # Predict
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (-1 / n_samples) * np.dot(X.T, (y - y_pred)) + self.alpha * np.sign(self.weights)
            db = (-1 / n_samples) * np.sum(y - y_pred)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Convergence check (L2 norm of weight updates)
            if np.linalg.norm(dw, ord=2) < self.tol:
                print(f"Converged at iteration {i+1}")
                break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias