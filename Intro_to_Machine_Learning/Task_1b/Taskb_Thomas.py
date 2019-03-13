import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import RidgeClassifierCV as ridgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')

# Retrieve data
dataval = data.values
len(dataval[0])

# Split in x and y data
xdata = dataval[:, 2:]
ydata = dataval[:, 1]

print (xdata)

# Initialize
Xvec = np.zeros((len(xdata), 21))

# Prepare data
for i in range(0, 21):
    idx = i % 5

    if i <= 4:
        Xvec[:, i] = xdata[:, idx]
    elif i <= 9:
        Xvec[:, i] = np.square(xdata[:, idx])
    elif i <= 14:
        Xvec[:, i] = np.exp(xdata[:, idx])
    elif i <= 19:
        Xvec[:, i] = np.cos(xdata[:, idx])
    else:
        Xvec[:, i] = 1

Xtrain = Xvec
Ytrain = ydata
Xtest = Xvec
Ytest = ydata


# prepare a range of ridge parameter values to test
upper_alpha = 500
lower_alpha = 1
step = 2
n_alpha = int((upper_alpha - lower_alpha) / step + 1)
alphas1 = np.linspace(lower_alpha, upper_alpha, n_alpha)

upper_alpha = 1
lower_alpha = 0.1
step = 0.05
n_alpha = int((upper_alpha - lower_alpha) / step + 1)
alphas2 = np.linspace(lower_alpha, upper_alpha, n_alpha)

alphas = [*alphas1, *alphas2]

# Initialize
upper_k = 10
lower_k = 2
step = 1
n_k1 = int((upper_k - lower_k) / step + 1)
k_range1 = np.linspace(lower_k, upper_k, n_k1)


k_range = k_range1

resultsRidge = np.zeros((len(k_range), 3))
resultsLasso = np.zeros((len(k_range), 3))

# Fit
cc = 0

k_range = [500, 1000]
for i in k_range:

    # create and fit a RIDGE regression model, testing each alpha
    modelRidge = ridge()
    gridRidge = GridSearchCV(cv=int(i), estimator=modelRidge,
                             param_grid=dict(alpha=alphas), iid=True)

    YmodelRidge = gridRidge.predict(Xtest)

    # create and fit a LASSO regression model, testing each alpha
    modelLasso = lasso()
    gridLasso = GridSearchCV(cv=int(i), estimator=modelLasso,
                             param_grid=dict(alpha=alphas), iid=True)
    gridLasso.fit(Xtrain, Ytrain)
    YmodelLasso = gridLasso.predict(Xtest)

    # RMSE
    RMSE_Ridge = mean_squared_error(YmodelRidge, Ytest) ** 0.5
    RMSE_Lasso = mean_squared_error(YmodelLasso, Ytest) ** 0.5

    # summarize the results of the grid search
    resultsRidge[cc, 0] = i
    resultsRidge[cc, 1] = gridRidge.best_estimator_.alpha
    resultsRidge[cc, 2] = RMSE_Ridge

    resultsLasso[cc, 0] = i
    resultsLasso[cc, 1] = gridLasso.best_estimator_.alpha
    resultsLasso[cc, 2] = RMSE_Lasso

    if cc == 0:
        bestRidgeRMSE = RMSE_Ridge
        bestRowRidge = resultsRidge[cc, :]

        bestLassoRMSE = RMSE_Lasso
        bestRowLasso = resultsLasso[cc, :]
        bestRidge = modelRidge
        bestLasso = modelLasso

    elif RMSE_Ridge < bestRidgeRMSE:
        bestRidgeRMSE = RMSE_Ridge
        bestRowRidge = resultsRidge[cc, :]
        bestRidge = modelRidge

    elif RMSE_Lasso < bestLassoRMSE:
        bestLassoRMSE = RMSE_Lasso
        bestRowLasso = resultsLasso[cc,:];
        bestLasso = modelLasso;

    cc = cc+1


np.set_printoptions(suppress=True)
print(bestRowRidge)
print(bestRowLasso)


# In[ ]:


# Retrieve best parameters
best_kRidge = bestRowRidge[0]
best_kLasso = bestRowLasso[0]
best_lambdaRidge = bestRowRidge[1]
best_lambdaLasso = bestRowLasso[1]

modelRidge = ridge(alpha=best_lambdaRidge)
modelRidge.fit(Xtrain,Ytrain);
YRidge = modelRidge.predict(Xtest)
RMSE_Ridge = mean_squared_error(YRidge, Ytest)**0.5

modelLasso = lasso(alpha=best_lambdaLasso)
modelLasso.fit(Xtrain,Ytrain);
YLasso = modelLasso.predict(Xtest)
RMSE_Lasso = mean_squared_error(YLasso, Ytest)**0.5

print(RMSE_Ridge)
print(RMSE_Lasso)


# In[ ]:


print(modelRidge.coef_)


# In[ ]:


print(modelLasso.coef_)
print(len(modelLasso.coef_))


# In[ ]:


df = pd.DataFrame(modelLasso.coef_)
df.to_csv("SolutionBayesWatch_v7.csv",index=False,header=False)

df

