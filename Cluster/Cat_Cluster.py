import pandas as pd
import pickle
import util
import re
from scipy import stats
import math
from sklearn import cluster, covariance, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math


def get_industry_descendant(industry_code='110000'):
    intern = util.get_all_pkl()
    stock_SWICS = intern['InstrumentInfo']['SWICS']
    pattern = re.compile(industry_code[0:2])
    descendant = []
    for index, stock in zip(intern['InstrumentInfo'].index, stock_SWICS):
        if re.match(pattern, stock):
            descendant.append(index)
    return np.asarray(descendant)


def get_stock_feature(stock_code=['000002'], feature='all'):
    intern = util.get_all_pkl()
    all_stock_index = intern['InstrumentInfo']
    temp = []
    if feature == 'all':
        for code in stock_code:
            stock_index = all_stock_index.index.get_loc(code)
            stock_all_feature = intern['MktData'].iloc[:, stock_index*9:(stock_index+1)*9]
            temp.append(stock_all_feature)
    else:
         for code in stock_code:
            stock_index = all_stock_index.index.get_loc(code)
            stock_all_feature = intern['MktData'].iloc[:, stock_index*9:(stock_index+1)*9]
            temp.append(stock_all_feature[feature])
    return pd.concat(temp,axis=1)


def get_stock_index(stock_code=['000001'], name=False):
    intern = util.get_all_pkl()
    all_stock_index = intern['InstrumentInfo']
    stock_index = [all_stock_index.index.get_loc(code_i) for code_i in stock_code]
    stock_name = [all_stock_index.loc[code_j, 'Name'] for code_j in stock_code]
    if not name:
        return np.asarray(stock_index)
    elif name:
        return np.asarray(stock_name)


def industry_corr_coef(industry_code='110000'):
    descendant = get_industry_descendant(industry_code)
    stock_name = get_stock_index(descendant, name=True)
    temp = get_stock_feature(descendant)
    ret_corr_coef = temp.corr(method='pearson')
    sns.heatmap(ret_corr_coef, annot=False, square=True, cmap='Blues', vmax=1)
    plt.show()

    

def industry_ret_corrcoef_cluster(industry_code='110000',name=False):
    descendant = get_industry_descendant(industry_code)
    stock_name = get_stock_index(descendant, name=True)
    temp = get_stock_feature(descendant,feature='ret')
    ret_corr_coef = temp.corr(method='pearson').fillna(0) 
    ap =  cluster.AffinityPropagation(preference=-1, damping=0.6)
    ap.fit(ret_corr_coef)
    labels = ap.labels_
    n_labels = labels.max()
#    print(ap.labels_)
    cluster_result = {} 
    if not name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = descendant[labels == i] 
    elif name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = stock_name[labels == i] 
            #print('Cluster %i: %s' % ((i + 1), ', '.join(descendant[labels == i])))
    return cluster_result
            
    
def industry_ret_cov_cluster(industry_code='110000', name=False):
    descendant = get_industry_descendant(industry_code)
    stock_name = get_stock_index(descendant, name=True)
    temp = get_stock_feature(descendant,feature='ret')
    ret_cov = temp.cov().fillna(0)
    _, labels = cluster.affinity_propagation(ret_cov)
    n_labels = labels.max()
    cluster_result = {}
    if not name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = descendant[labels == i]
            #print('Cluster %i: %s' % ((i + 1), ', '.join(stock_name[labels == i])))
    elif name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = stock_name[labels == i]
            #print('Cluster %i: %s' % ((i + 1), ', '.join(descendant[labels == i])))
    return cluster_result


def industry_ret_corrcoef_kmeans(industry_code='110000',name=False):
    descendant = get_industry_descendant(industry_code)
    stock_name = get_stock_index(descendant, name=True)
    temp = get_stock_feature(descendant,feature='ret')
    ret_cov = temp.cov().fillna(0)
    clf = preprocessing.Normalizer()
    ret_cov = clf.fit_transform(ret_cov)
#    loss = []
#    for k in range(2, 20):
#        clf = cluster.KMeans(n_clusters=k)
#        clf.fit(ret_cov)
#        loss.append(clf.inertia_)
#    fig = plt.figure(figsize=(15, 5))
#    plt.plot(range(2, 20), loss)
#    plt.grid(True)
#    plt.title("Loss curve versus cluster centers ")
#    plt.show()    
    clf = cluster.KMeans(n_clusters=4)
    clf.fit(ret_cov)
    labels = clf.labels_
    n_labels = labels.max()
    cluster_result = {}
    if not name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = descendant[labels == i] 
            #print('Cluster %i: %s' % ((i + 1), ', '.join(stock_name[labels == i])))
    elif name:
        for i in range(n_labels + 1):
            cluster_result[i+1] = stock_name[labels == i] 
            #print('Cluster %i: %s' % ((i + 1), ', '.join(descendant[labels == i])))
    return cluster_result

    
def plot_cluster_heatmap(cluster=['000719','002247','002261','002517', '600637', '600832'], annot=False, curve=False):
    ret = get_stock_feature(stock_code=cluster, feature='ret')
    ret_corr_coef = ret.corr(method='pearson')
    fig = plt.figure( figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1,2,1)
    ax1.set(title='Correlation coefficient between stocks')
    sns.heatmap(ret_corr_coef, annot=annot, square=True, cmap='Blues', vmax=1 )
    if curve:
        ax2 = fig.add_subplot(1,2,2)
        ax2.set(title='Daily return')
        ax2.set_ylabel('Daily return')
        ax2.set_xlabel('Trading date')
        ax2.set_xticks([])
        for stock in ret.columns:
            ax2.plot(ret[stock], label = stock)
        ax2.legend(loc='upper right')   
    plt.show()
    return ret_corr_coef
    
#
#def split_stock_data(stock_code=['000001']):
#    ret_train = []
#    ret_test = []
#    for stock in stock_code:
#        
#    return 
    

if __name__ == '__main__':
    cluster_result = industry_ret_corrcoef_cluster(industry_code='110000', name=False)
    plot_cluster_heatmap(cluster_result[8], annot=True, curve=True)
    
    

    