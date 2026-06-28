# src/linear_regression.py
from sklearn.linear_model import LinearRegression

class HouseLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)