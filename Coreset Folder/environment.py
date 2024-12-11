x = 1

print(x)


# Import the necessary packages
import numpy as np
import pandas as pd

import sklearn


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpg = pd.read_csv('diabetes.csv')

# User-defined input feature1
X = mpg[["Glucose"]]

# Output feature: mpg
y = mpg[['Outcome']]  # No changes here

# Create training and testing data with 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Changes start here
# Reshape y_train and y_test for scaling
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Scale the target variable
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Flatten y_train and y_test after scaling for SVR fitting
y_train = y_train.ravel()  # Changes here
y_test = y_test.ravel()    # Changes here

# Initialize and fit a linear SVR model to training data with epsilon = 0.2
eps = 0.2
svr_lin = SVR(kernel='linear', epsilon=eps)
svr_lin.fit(X_train, y_train)

# Initialize and fit an SVR model using a poly kernel with epsilon=0.2, C=0.5, and gamma=0.7
svr_poly = SVR(kernel='poly', epsilon=0.2, C=0.5, gamma=0.7)
svr_poly.fit(X_train, y_train)

# Initialize and fit an SVR model using an RBF kernel with epsilon=0.2, C=0.5, and gamma=0.7
svr_rbf = SVR(kernel='rbf', epsilon=0.2, C=0.5, gamma=0.7)
svr_rbf.fit(X_train, y_train)

# Print the coefficients of determination for each model
lin_score = svr_lin.score(X_test, y_test)
print('Linear model:', np.round(lin_score, 3))

poly_score = svr_poly.score(X_test, y_test)
print('Polynomial model:', np.round(poly_score, 3))

rbf_score = svr_rbf.score(X_test, y_test)
print('RBF model:', np.round(rbf_score, 3))
