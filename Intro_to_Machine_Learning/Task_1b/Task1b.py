import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso as Lasso

import csv


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size,
                 chunk_size)


def ReadFile():
    data = pd.read_csv('train.csv')
    return data


def x2Phi(data):
    data = data.values

    phi0 = data[:, 2:]
    phi1 = np.square(data[:, 2:])
    phi2 = np.exp(data[:, 2:])
    phi3 = np.cos(data[:, 2:])
    # phi4 = np.ones((len(data), 1))

    phi = np.concatenate((phi0, phi1, phi2, phi3), axis=1)
    y = data[:, 1]

    return phi, y


def SplitData(data, y, chunkSize):
    indices = index_marks(int(len(data)), chunkSize)
    return np.split(data, indices), np.split(y, indices)


def Train(data, y, kFold, lambd):
    XTrain = np.empty((0, 20))
    YTrain = np.empty((0, ))

    for i in range(0, 100):
        if i == kFold:
            continue
        XTrain = np.append(XTrain, data[i][:, :], axis=0)
        YTrain = np.append(YTrain, y[i], axis=0)

    XTest = data[kFold][:, :]
    YTest = y[kFold]

    model = Lasso(alpha=lambd, fit_intercept=True)
    model.fit(XTrain, YTrain)

    YPred = model.predict(XTest)
    RMSE = mean_squared_error(YTest, YPred) ** 0.5
    scores = model.coef_

    return RMSE, scores, model.intercept_


def main():
    kFolds = 100
    # lambdas = list(range(1, 100))
    lambdas = [0.17, 0.171, 0.172, 0.173]

    RMSE = np.zeros((len(lambdas), ))
    inter = np.zeros((len(lambdas), ))
    weights = np.zeros((len(lambdas), 20))

    data = ReadFile()
    phi, y = x2Phi(data)
    splitData, splitY = SplitData(phi, y, int(len(phi) / kFolds))

    for l in range(len(lambdas)):
        for k in range(kFolds):
            r, w, inters = Train(splitData, splitY, k, lambdas[l])
            RMSE[l] = RMSE[l] + r
            inter[l] = inter[l] + inters
            weights[l, :] = weights[l, :] + w

    RMSE = RMSE / kFolds
    inter = inter / kFolds
    weights = weights / kFolds
    val, idx = min((val, idx) for (idx, val) in enumerate(RMSE))
    bestWeights = weights[idx, :]
    print (inter[idx])
    print (type(bestWeights))
    bestWeights = np.append(bestWeights, inter[idx])

    print (bestWeights)
    print(idx)
    print (RMSE[idx])

    with open('ParkerTest2.csv', "w") as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow(bestWeights)


if __name__ == '__main__':
    main()
