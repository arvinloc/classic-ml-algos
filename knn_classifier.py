import numpy as np
from collections import Counter
def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x2 - x1) ** 2))


class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predicted_values = [self._predict(x) for x in X]
        return predicted_values

    def _predict(self,x):

        distances = [euclidian_distance(x,x_train) for x_train in self.X_train]

        k_idxs = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idxs]

        result = Counter(k_nearest_labels).most_common(1)

        return result[0][0]



knn = KNN()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X,y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)


X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


knn.fit(X_train,y_train)

predictions = knn.predict(X_test)
#print(predictions)
acc = np.sum(predictions == y_test) / len(y_test)
print(f"accuracy: {acc}")