import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statsmodels
from statsmodels.tsa.stattools import coint
from sklearn import cluster, covariance, manifold, preprocessing
from matplotlib.collections import LineCollection
import pickle


# 获取股票代码
intern = pickle.load(open(r'D:/Obsidia/Data/intern.pkl', 'rb'))
df_cap = intern['MktCap']
index = np.arange(1, 1705)
np.random.shuffle(index)
names = np.array(df_cap.columns[index][:100])
# 获取open, close并计算var
df_data = intern['MktData']
open_price = df_data.iloc[:, [(i-1)*9+1 for i in index]]
close_price = df_data.iloc[:, [(j-1)*9+4 for j in index]]
open_price_full = open_price.fillna(open_price.mean()).as_matrix()
close_price_full = close_price.fillna(close_price.mean()).as_matrix()
print("Calcaulating the var ")
var = open_price_full - close_price_full
edge_model = covariance.GraphLassoCV()
X = var/var.std(0)
# var_scaler = preprocessing.StandardScaler()
# X = var_scaler.fit_transform(X)
print("Training the covariance matrix")
edge_model.fit(X)
print("Training the clustering center")
centers, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))



# Visualization
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)
embedding = node_position_model.fit_transform(X.T).T
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)
# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))
plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())
plt.show()
