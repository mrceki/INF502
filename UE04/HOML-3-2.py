import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.ndimage import shift


def shift_image(image, direction):
    if isinstance(image, str):                        #dataset yüklenirken 'as_frame' parametresi false verilmezste str olarak bir data geliyor. 
        image = np.fromstring(image, dtype=np.uint8)
    
    image = image.reshape((28, 28))
    
    if direction == 'left':
        shifted_image = shift(image, [0, -1], cval=0)
    elif direction == 'right':
        shifted_image = shift(image, [0, 1], cval=0)
    elif direction == 'up':
        shifted_image = shift(image, [-1, 0], cval=0)
    elif direction == 'down':
        shifted_image = shift(image, [1, 0], cval=0)
    else:
        raise ValueError("Invalid direction. Must be one of 'left', 'right', 'up', or 'down'.")
    return shifted_image.ravel()

# Load dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False) # as_frame false verilmezse img string olarak geliyor
X, y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create copies
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]
try:
    print("Augmention started")
    for image, label in zip(X_train, y_train):
        for direction in ['left', 'right', 'up', 'down']:
            X_train_augmented.append(shift_image(image, direction))
            y_train_augmented.append(label)
    print("Augmention finished")
except:
    print("Augmention failed!!!")

# Convert dataset back
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

"""

TRAIN WITH AUGMENTED DATASET

""" 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_train_augmented, y_train_augmented, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [2,3,4], 'weights': ['uniform', 'distance']}

try:    
    print("Grid search started")
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train_aug, y_train_aug)
    print("Grid search finished")
except:
    print("Grid search failed!!!!")

print("Best hyperparameters are:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], 
                                weights=grid_search.best_params_['weights'])
best_knn.fit(X_train_aug, y_train_aug)

test_score = best_knn.score(X_test_aug, y_test_aug)
print("Test accuracy score:", test_score)

