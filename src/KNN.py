import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            # 计算欧氏距离 (L2 Distance)
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_indices]
            counts = np.bincount(k_labels.astype(int))
            preds.append(np.argmax(counts))
        return np.array(preds)