# src/SVM.py
from sklearn.svm import SVC

class TitanicSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        # 使用径向基核（RBF），通常在非线性分类问题中表现最优
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)