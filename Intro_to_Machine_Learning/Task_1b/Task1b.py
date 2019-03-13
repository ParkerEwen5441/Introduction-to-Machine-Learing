import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso as Lasso
from sklearn.feature_selection import SelectKBest, chi2

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
    phi4 = np.ones((len(data), 1))

    phi = np.concatenate((phi0, phi1, phi2, phi3, phi4), axis=1)

    y = data[:, 1]

    return phi, y


def SplitData(data, y, chunkSize):
    indices = index_marks(int(len(data)), chunkSize)
    return np.split(data, indices), np.split(y, indices)


def Train(data, y, kFold, lambd):
    XTrain = np.empty((0, 21))
    YTrain = np.empty((0, ))

    for i in range(0, 10):
        if i == kFold:
            continue
        XTrain = np.append(XTrain, data[i][:, :], axis=0)
        YTrain = np.append(YTrain, y[i], axis=0)

    XTest = data[kFold][:, :]
    YTest = y[kFold]

    # selector = SelectKBest(chi2, k='all').fit(XTrain, YTrain)
    # X_new = selector.transform(XTrain)
    model = Lasso(alpha=lambd, fit_intercept=False)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    RMSE = mean_squared_error(YTest, YPred) ** 0.5
    scores = model.coef_
    print (model.intercept_)

    return RMSE, scores


def main():
    RMSE = np.empty((5, 1))
    weights = np.empty((5, 21))

    kFolds = 10
    lambdas = [0.1, 1, 10, 100, 1000]

    data = ReadFile()
    phi, y = x2Phi(data)
    splitData, splitY = SplitData(phi, y, int(len(phi) / kFolds))

    # X_train,X_test,y_train,y_test=train_test_split(splitData,splitY,test_size=0.1)

    for l in range(len(lambdas)):
        for k in range(kFolds):
            r, w = Train(splitData, splitY, k, lambdas[l])
            RMSE[l] = RMSE[l] + r
            weights[l] = weights[l] + w
        RMSE[l] = RMSE[l] / kFolds
        weights[l] = weights[l] / kFolds

    val, idx = min((val, idx) for (idx, val) in enumerate(RMSE))
    bestWeights = weights[idx, :]

    print (bestWeights)
    with open('ParkerTest2.csv', "w") as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow(bestWeights)


if __name__ == '__main__':
    main()
