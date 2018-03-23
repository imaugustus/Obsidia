import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statsmodels
from statsmodels.tsa.stattools import coint
from sklearn.cluster import KMeans
import pickle

## 数据处理，找到缺失值
intern = pickle.load(open(r'D:/Obsidia/Data/intern.pkl', 'rb'))
df = intern['MktCap']
df_not_null = df.fillna(df.mean())#用每个Series的平均值数据填充空缺值
# 按照股票总市值的均值和标准差进行K-means聚类
X_train = pd.DataFrame({"mean": df_not_null.mean(), "std": df_not_null.std()}).as_matrix()
estimator = KMeans(n_clusters=3)
estimator.fit(X_train)
# print(estimator.cluster_centers_) 输出聚类的中心
df_result = pd.DataFrame(pd.DataFrame({"mean": df_not_null.mean(), "std": df_not_null.std(), "Labels": estimator.labels_}))
# 按照每支股票的分类画出其分类散点图
mean = df_not_null.mean().as_matrix()
std = df_not_null.std().as_matrix()
labels = estimator.labels_
color = ['r', 'b', 'g']
for i in range(len(mean)):
    plt.scatter(x=mean[i], y=std[i], c=color[labels[i]])
plt.xlabel("MeanPrice")
plt.ylabel("StdPrice")
plt.title("Classification of Stocks By Mean and Std")
plt.show()























## 按照股票的总市值进行聚类
# clf = KMeans(n_clusters=5)
# clf.fit(df_not_null.mean().reshape(-1, 1))
# mean_labels = clf.labels_
# print(clf.cluster_centers_)
# color = ['r', 'b', 'g', 'c', 'y']
# for i in range(len(mean)):
#     plt.scatter(x=1, y=mean[i], c=color[mean_labels[i]])
# plt.show()
## 按照每只股票的波动率进行聚类
