import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statsmodels
from statsmodels.tsa.stattools import coint

np.random.seed(100)
x = np.random.normal(0, 1, 600)
# print(x.shape)
# print(np.cumsum(x).shape)
y = np.random.normal(0, 1, 600)
X = pd.Series(np.cumsum(x)) + 100
# print(X)
Y = X + y + 30
for i in range(600):
    X[i] = X[i] - i/10
    Y[i] = Y[i] - i/10

plt.plot(pd.DataFrame({'X':X, 'Y':Y}))
plt.plot(pd.DataFrame({'Y-X': Y-X, 'X': np.mean(Y-X)}))
plt.show()