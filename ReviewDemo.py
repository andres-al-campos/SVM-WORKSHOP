# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:25:00 2018

@author: Sunny
"""


from sklearn.svm import SVC,SVR
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [1, 1]]
y = [0, 1]

logReg = linear_model.LogisticRegression(C=1e5)
logReg.fit(X,y)
print('Classification:--')
print(logReg.predict([[2., 2.]]))

supVecC = SVC()
supVecC.fit(X, y)

print(supVecC.predict([[2., 2.]]))


print('Regression:--')


X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = linear_model.LinearRegression()
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
