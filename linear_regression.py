import numpy as np

class LinearRegression:
    def __init__(self,l_rate = 0.001,n_iters = 1000):
        self.l_rate = l_rate
        self.n_iters = n_iters


    def fit(self,X,y):
        n_samples,n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iters):
            y_predicted = np.dot(X,self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T,(y_predicted - y))
            db =  (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.l_rate * dw
            self.bias -= self.l_rate * db


    def predict(self,X):
        return np.dot(X,self.weights) + self.bias



from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split

X,y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=20,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
)


print(X.shape)
print(y.shape)

regression = LinearRegression()

regression.fit(X_train,y_train)

y_pred = regression.predict(X_test)

print(f"mse: {np.mean((y_test - y_pred)**2)}")
