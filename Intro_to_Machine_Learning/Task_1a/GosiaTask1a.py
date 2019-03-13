# Intro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import random
import csv

from sklearn.model_selection import train_test_split as Split
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Ridge as Ridge
from sklearn.metrics import mean_squared_error

with open('train.csv') as file:
    data = list(csv.reader(file))

# Split
X = np.zeros((len(data),10))
Y = np.zeros((len(data),1))
Id = np.zeros((len(data),1))
counter = 0
#np.random.shuffle(data[1:])
for i in range(len(data)): # for every row (data set)
    if i == 0:
        counter += 1 # skip first row (titles)
    else:
        for j in range(10):
            X[i][j] = float(data[i][j+2])
        Y[i] = float(data[i][1])
        Id[i] = int(data[i][0])
        counter += 1

# Plot Data
#for i in range(10):
#    plt.scatter(Id[:], X[:,i],  color='green', marker='x')
#plt.scatter(Id[:], Y[:], color='black')

# 10-fold split
folds = 10
fold_len = int((len(data)-1)/folds)
features = 10
X_fold = np.zeros((folds,fold_len,features))
Y_fold = np.zeros((folds,fold_len))
# X_fold: fold No, position in fold, feature
for i in range(folds):
    for j in range(fold_len):
        X_fold[i,j,:] = X[i*fold_len + j + 1][:]
        Y_fold[i,j] = Y[i*fold_len + j + 1]

# Cross-validation
L = [0.1,1,10,100,1000] # lambdas
RMSE = np.zeros(len(L))
for l in range(len(L)):
    for k in range(folds):
        X_val = X_fold[k,:,:]
        Y_val = Y_fold[k,:]
        X_train_3d = np.concatenate((X_fold[:k,:,:], X_fold[k+1:,:,:]))
        X_train = np.reshape(X_train_3d, ((folds-1)*fold_len, features))
        Y_train_3d = np.concatenate((Y_fold[:k,:], Y_fold[k+1:,:]))
        Y_train = np.reshape(Y_train_3d, ((folds-1)*fold_len, 1))
        model = Ridge(alpha=L[l])
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_val)
        RMSE[l] = RMSE[l] + mean_squared_error(Y_val, Y_pred)**0.5 #summation
    RMSE_avg = RMSE/folds #average
#print(RMSE_avg)

# Write into CSV
with open('output.csv', mode='w', newline='') as output_file:
    output_writer = csv.writer(output_file, delimiter=',')
    for i in range(len(RMSE_avg)):
        output_writer.writerow([RMSE_avg[i]])
