from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

k_range = list(range(1, 50))
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors=k_range, weights=weight_options)
# print (param_grid)
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)


print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

K = grid.best_params_['n_neighbors']
weights = grid.best_params_['weights']
knn = KNeighborsClassifier(n_neighbors=K, weights=weights)
knn.fit(X, y)

TEST_X = [[1.2, 2.2, 3.6, 5.2]]
ind, dist = knn.kneighbors(TEST_X, n_neighbors=K)
print(knn.predict(TEST_X))
print(dist)
print(ind)