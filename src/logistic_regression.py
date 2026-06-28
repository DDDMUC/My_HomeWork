import numpy as np

class LogisticRegressionCustom:
    def __init__(self, lr=0.1, lambda_reg=0.1, epochs=100, batch_size=32):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_s = X[indices]
            y_s = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                Xi = X_s[i : i + self.batch_size]
                yi = y_s[i : i + self.batch_size]
                m = Xi.shape[0]
                if m == 0: continue
                
                pred = self.sigmoid(np.dot(Xi, self.w) + self.b)
                dw = (1/m) * np.dot(Xi.T, (pred - yi)) + (self.lambda_reg/m) * self.w
                db = (1/m) * np.sum(pred - yi)
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            # 计算每轮损失
            full_pred = self.sigmoid(np.dot(X, self.w) + self.b)
            full_pred = np.clip(full_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(full_pred) + (1 - y) * np.log(1 - full_pred)) \
                   + (self.lambda_reg / (2 * n_samples)) * np.sum(self.w ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        proba = self.sigmoid(np.dot(X, self.w) + self.b)
        return (proba >= 0.5).astype(int)