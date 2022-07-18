#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from mat2json import loadMat
from util import getBatteryCapacity, getChargingValues, getDischargingValues, getDataframe, series_to_supervised, rollingAverage
from tabulate import tabulate


# In[21]:


print('\033[1m' + "Battery B0005\n" + '\033[0m')

B0005 = loadMat('B0005.mat')
dfB0005 = getDataframe(B0005)
print(dfB0005.head())


X = dfB0005.drop('capacity', axis=1)
Y = dfB0005['capacity']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

best_svr = SVR(C=1, epsilon=0.0001, gamma=0.00001, cache_size=10, kernel='rbf', max_iter=1000, shrinking=True, tol=0.001, verbose=False)
poly_svr=SVR(epsilon=0.1, gamma="auto", kernel='poly', shrinking=True, tol=0.001, verbose=False, coef0=1,degree=3)
linear_svr=SVR(kernel="linear",gamma="auto",C=1)

best_svr.fit(X_train, y_train)
poly_svr.fit(X_train,y_train)
linear_svr.fit(X_train,y_train)

# rbf kernel SVR
y_pred = best_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("rbf kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", best_svr.score(X_test, y_test))

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0005 using rbf Kernel')
ax.legend()

# poly kernel SVR
y_pred = poly_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\npoly kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", poly_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0005 using Poly Kernel')
ax.legend()

# linear kernel SVR
y_pred = linear_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\nlinear kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", linear_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0005 using Linear Kernel')
ax.legend()


# In[18]:


print('\033[1m' + "Battery B0006\n" + '\033[0m')

B0006 = loadMat('B0006.mat')
dfB0006 = getDataframe(B0006)
dfB0006.head()


X = dfB0006.drop('capacity', axis=1)
Y = dfB0006['capacity']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

best_svr = SVR(C=1, epsilon=0.0001, gamma=0.00001, cache_size=10, kernel='rbf', max_iter=1000, shrinking=True, tol=0.001, verbose=False)
poly_svr=SVR(epsilon=0.1, gamma="auto", kernel='poly', shrinking=True, tol=0.001, verbose=False, coef0=1,degree=3)
linear_svr=SVR(kernel="linear",gamma="auto",C=1)

best_svr.fit(X_train, y_train)
poly_svr.fit(X_train,y_train)
linear_svr.fit(X_train,y_train)


# rbf kernel SVR
y_pred = best_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("rbf kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", best_svr.score(X_test, y_test))

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0006 using rbf Kernel')
ax.legend()

# poly kernel SVR
y_pred = poly_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\npoly kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", poly_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0006 using Poly Kernel')
ax.legend()

# linear kernel SVR
y_pred = linear_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\nlinear kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", linear_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0006 using Linear Kernel')
ax.legend()


# In[19]:


print('\033[1m' + "Battery B0007\n" + '\033[0m')

B0007 = loadMat('B0007.mat')
dfB0007 = getDataframe(B0007)
dfB0007.head()


X = dfB0007.drop('capacity', axis=1)
Y = dfB0007['capacity']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

best_svr = SVR(C=1, epsilon=0.0001, gamma=0.00001, cache_size=10, kernel='rbf', max_iter=1000, shrinking=True, tol=0.001, verbose=False)
poly_svr=SVR(epsilon=0.1, gamma="auto", kernel='poly', shrinking=True, tol=0.001, verbose=False, coef0=1,degree=3)
linear_svr=SVR(kernel="linear",gamma="auto",C=1)

best_svr.fit(X_train, y_train)
poly_svr.fit(X_train,y_train)
linear_svr.fit(X_train,y_train)


# rbf kernel SVR
y_pred = best_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("rbf kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", best_svr.score(X_test, y_test))

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0007 using rbf Kernel')
ax.legend()

# poly kernel SVR
y_pred = poly_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\npoly kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", poly_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0007 using Poly Kernel')
ax.legend()

# linear kernel SVR
y_pred = linear_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\nlinear kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", linear_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B0007 using Linear Kernel')
ax.legend()


# In[20]:


print('\033[1m' + "Battery B0018\n" + '\033[0m')

B0018 = loadMat('B0018.mat')
dfB0018 = getDataframe(B0018)
dfB0018.head()

X = dfB0018.drop('capacity', axis=1)
Y = dfB0018['capacity']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

best_svr = SVR(C=1, epsilon=0.0001, gamma=0.00001, cache_size=10, kernel='rbf', max_iter=1000, shrinking=True, tol=0.001, verbose=False)
poly_svr=SVR(epsilon=0.1, gamma="auto", kernel='poly', shrinking=True, tol=0.001, verbose=False, coef0=1,degree=3)
linear_svr=SVR(kernel="linear",gamma="auto",C=1)

best_svr.fit(X_train, y_train)
poly_svr.fit(X_train,y_train)
linear_svr.fit(X_train,y_train)


# rbf kernel SVR
y_pred = best_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("rbf kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", best_svr.score(X_test, y_test))

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B00018 using rbf Kernel')
ax.legend()

# poly kernel SVR
y_pred = poly_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\npoly kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", poly_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B00018 using Poly Kernel')
ax.legend()

# linear kernel SVR
y_pred = linear_svr.predict(X_test)


scores_regr = metrics.mean_squared_error(y_test, y_pred)
print("\nlinear kernel")
print("Mean Squared Error is", scores_regr)
print("Accuracy is", linear_svr.score(X_test, y_test))


fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X_test['cycle'], y_test, color='black', label='Battery Capacity')
ax.plot(X_test['cycle'], y_pred, color='red', label='Prediction')

ax.set(xlabel='Cycle', ylabel="Capacity", title='Model performance for Battery B00018 using Linear Kernel')
ax.legend()


# In[ ]:




