import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, covariance, manifold
import seaborn as sb
import math

## 计算股票收益平均和波动率
intern = pickle.load(open(r'D:/Obsidia/Data/intern.pkl', 'rb'))
stock_count = len(intern['MktCap'].columns)
# price_count = len(intern['MtkData'].columns)
close_price_index = [i*9 for i in range(stock_count)]
close = intern['MktData'].iloc[:, close_price_index]
close = close.fillna(close.mean())
returns = close.pct_change().mean()*803
volatility = close.pct_change().std() * math.sqrt(803)
data = pd.concat([returns, volatility], axis=1)

## 计算不同聚类中心数下的聚类损失函数以找到最合适的聚类中心数目
loss = []
for k in range(20, 40):
    clf = cluster.KMeans(n_clusters=k)
    clf.fit(data)
    loss.append(clf.inertia_)

fig = plt.figure(figsize=(40, 10))
plt.plot(range(20, 40), loss)
plt.grid(True)
plt.title("Loss curve versus cluster centers ")
# plt.show()

## 按照最终分类聚类(Kmeans)
final_cluster = cluster.KMeans(n_clusters=40)
final_cluster.fit(data)
category = {}
for label in range(40):
    category["cluster"+str(label)] = intern['MktCap'].columns[final_cluster.labels_ == label]

for code in category['cluster39']:
    print(code)
    industry, name = intern['InstrumentInfo'].ix[code, [0, 2]]
    print(industry, name)

# ## 采用covariance分类
# edge_model = covariance.GraphLassoCV()
# edge_model.fit(data)
# centers, labels = cluster.affinity_propagation(edge_model.covariance_)
# n_labels = labels.max()
# category_nd = {}
# for label in range(n_labels + 1):
#     category_nd["cluster_nd" + str(label)] = intern['MktCap'].columns[labels == label]
#
# print(category_nd)