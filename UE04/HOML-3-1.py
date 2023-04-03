from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
mnist = fetch_openml('mnist_784', version=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# Create KNeighborsClassifier
knn = KNeighborsClassifier()

# Define hyperparameters for grid search
param_grid = { 'weights': ['uniform', 'distance'], 'n_neighbors': [3, 5, 7, 9]}

# Find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print results
print("Best hyperparameters are:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# Create a KNeighborsClassifier object with the best hyperparameters
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], 
                                weights=grid_search.best_params_['weights'])

# Fit the model
best_knn.fit(X_train, y_train)
test_score = best_knn.score(X_test, y_test)

# Print score
print("Test accuracy score:", test_score)
