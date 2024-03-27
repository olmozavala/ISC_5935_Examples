import numpy as np
from sklearn.model_selection import train_test_split
X, y,indices = (0.1*np.arange(10)).reshape((5, 2)),range(10,15),range(5)
X_train, X_test, y_train, y_test, indices_train,indices_test = train_test_split(X, y,indices, test_size=0.33, random_state=42)
indices_train,indices_test