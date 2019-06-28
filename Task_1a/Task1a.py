import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge as KernelRidge

import csv


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size,
                 chunk_size)


def ReadFile():
    data = pd.read_csv('train.csv')
    return data


def SplitData(data, chunkSize):
    indices = index_marks(data.shape[0], chunkSize)
    return np.split(data, indices)


def Train(data, kFold, lambd):
    RMSE = 0
    x_start = 2
    x_end = 12

    XTrain = pd.DataFrame().reindex_like(data[0].iloc[:, x_start:x_end]) * 0
    YTrain = pd.DataFrame([0])

    for i in range(0, 10):
        if i == kFold:
            continue
        X = data[i].iloc[:, x_start:x_end]
        Y = data[i].iloc[:, 1]
        XTrain = pd.concat([XTrain, X], sort=False, ignore_index=True)
        YTrain = pd.concat([YTrain, Y], sort=False, ignore_index=True)

    XTrain = XTrain.iloc[50:, ]
    YTrain = YTrain.iloc[1:, ]

    XTest = data[kFold].iloc[:, x_start:x_end]
    YTest = data[kFold].iloc[:, 1]

    model = KernelRidge(alpha=lambd)
    model.fit(XTrain, YTrain)

    YPred = model.predict(XTest)
    RMSE = RMSE + mean_squared_error(YTest, YPred) ** 0.5

    return RMSE


def main():
    kFolds = 10
    lambdas = [0.1, 1, 10, 100, 1000]

    data = ReadFile()
    splitData = SplitData(data, int(len(data.index) / kFolds))

    RMSE = [0] * len(lambdas)

    for lambd in range(len(lambdas)):
        for fold in range(0, 10):
            RMSE[lambd] = RMSE[lambd] + Train(splitData, fold, lambdas[lambd])
        RMSE[lambd] = (RMSE[lambd] / kFolds)

    with open('ParkerTestkernel2.csv', "w") as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow(RMSE)


if __name__ == '__main__':
    main()
