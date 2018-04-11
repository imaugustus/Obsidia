import numpy as np
import pandas as pd
import Strategy.ARMA_Strategy
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import re
import math
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import mstats

intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
MktData = MktData.swaplevel(0, 1, axis=1)
factor = pickle.load(open(r'D:/sync/Factor/v4/factor_predict_ts.pkl', 'rb'))
factor = factor.dropna(axis=1, how='all')
# factor_real = pickle.load(open(r'D:/sync/Factor/v4/factor_train_ts.pkl', 'rb'))
# factor_real = factor.dropna(axis=1, how='all')


class Group:
    def __init__(self, industry_code):
        self.industry_code = industry_code

    # 中位数去极值
    def filter_extreme(self, factor_section, n=5):
        Dm = factor_section.median()
        Dm1 = ((factor_section - Dm).abs()).median()
        max_limit = Dm + n * Dm1
        min_limit = Dm - n * Dm1
        factor_section = np.clip(factor_section, min_limit, max_limit)
        return factor_section

    # 标准化
    def normalize(self, factor_section):
        mean = factor_section.mean()
        std = factor_section.std()
        factor_section = (factor_section - mean) / std
        return factor_section

    # 缺失值处理
    def fill_na(self, factor_section):
        factor_section = factor_section.fillna(0)
        return factor_section

    # 预处理因子
    def preprocess_factor(self, factor):
        preprocessed_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
        for date in factor.index:
            section_factor = factor.loc[date, :]
            section_factor = self.filter_extreme(section_factor)
            section_factor = self.normalize(section_factor)
            section_factor = self.fill_na(section_factor)
            preprocessed_factor.loc[date, :] = section_factor
        return preprocessed_factor

    # 对每个截面按照分层结果画每个group的累计收益图
    def plot_group_cumsum_ret(self, ts):
        group_ret_ts = pd.DataFrame(index=ts.index, columns=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'])
        for date in ts.index:
            gb = ts.loc[date].groupby(pd.cut(ts.loc[date], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
            for group_label in gb.groups.keys():
                group_stock = list(gb.groups[group_label])
                group_ret_mean = MktData.loc[date, (group_stock, 'ret')].mean()
                group_ret_ts.loc[date, group_label] = group_ret_mean
        group_ret_ts.cumsum().plot()
        plt.show()
        return group_ret_ts.cumsum()

    # 对每个截面按照分层结果画每个group的累计超额收益图
    def plot_group_extra_performance(self, ts):
        group_extra_performance_ts = pd.DataFrame(index=ts.index, columns=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'])
        for date in ts.index:
            gb = ts.loc[date].groupby(pd.cut(ts.loc[date], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
            gb_index = MktData.loc[date, (list(ts.columns), 'ret')].mean()
            for group_label in gb.groups.keys():
                group_stock = list(gb.groups[group_label])
                group_extra_performance = MktData.loc[date, (group_stock, 'ret')].mean() - gb_index
                group_extra_performance_ts.loc[date, group_label] = group_extra_performance
        group_extra_performance_ts.cumsum().plot()
        plt.show()
        return group_extra_performance_ts.cumsum()


class Regression:
    def __init__(self, industry_code):
        self.industry_code = industry_code

# 中位数去极值
    def filter_extreme(self, factor_section, n=5):
        Dm = factor_section.median()
        Dm1 = ((factor_section-Dm).abs()).median()
        max_limit = Dm + n*Dm1
        min_limit = Dm - n*Dm1
        factor_section = np.clip(factor_section, min_limit, max_limit)
        return factor_section

# 标准化
    def normalize(self, factor_section):
        mean = factor_section.mean()
        std = factor_section.std()
        factor_section = (factor_section - mean)/std
        return factor_section

# 缺失值处理
    def fill_na(self, factor_section):
        factor_section = factor_section.fillna(0)
        return factor_section

# # 获取截面期因子暴露以及下个截面期收益
#     def get_train_data(self, factor_ts):
#         stock = list(factor_ts.columns)
#         all_section_factor_x = []
#         all_section_factor_y = []
#         for section in range(0, factor_ts.shape[0], 32):
#             section_factor_x = factor_ts.iloc[section, :]
#             all_section_factor_x.append(section_factor_x)
#             section_date = factor_ts.index[section]
#             # print("Factor section date:", section_date)
#             index = MktData.index.get_loc(section_date)
#             ret_start_index = index + 1
#             ret_end_index = index + 31
#             ret = MktData.loc[MktData.index[ret_start_index:ret_end_index], (stock, 'ret')]
#             section_factor_y = ret.mean(axis=0)
#             section_factor_y.index = section_factor_y.index.droplevel(1)
#             all_section_factor_y.append(section_factor_y)
#             # ret_start_date = MktData.index[ret_start_index]
#             # ret_end_date = MktData.index[ret_end_index]
#             # print('ret start date:', ret_start_date)
#             # print('ret end date:', ret_end_date)
#         return all_section_factor_x, all_section_factor_y

# 获取截面期因子暴露以及下个截面期收益
    def get_train_data(self, factor_ts):
        stock = list(factor_ts.columns)
        all_section_factor_x = []
        all_section_ret_y = []
        for section in range(0, factor_ts.shape[0]):
            section_factor_x = factor_ts.iloc[section, :]
            all_section_factor_x.append(section_factor_x)
            section_date = factor_ts.index[section]
            section_date_index = MktData.index.get_loc(section_date)
            ret_date_index = section_date_index + 1
            try:
                ret = MktData.loc[MktData.index[ret_date_index], (stock, 'ret')]
                ret.index = ret.index.droplevel(level=1)
                all_section_ret_y.append(ret)
            except KeyError:
                continue
        return all_section_factor_x, all_section_ret_y

# 预处理因子
    def preprocess_factor(self, all_section_factor_x):
        preprocessed_all_section_factor_x = []
        for section_factor in all_section_factor_x:
            section_factor = self.filter_extreme(section_factor)
            section_factor = self.normalize(section_factor)
            section_factor = self.fill_na(section_factor)
            preprocessed_all_section_factor_x.append(section_factor)
        return preprocessed_all_section_factor_x

#线性回归以求得因子系数(因子载荷)
    def get_factor_load(self):
        all_section_factor_x, all_section_factor_y = self.get_train_data(factor)
        preprocessed_all_section_factor_x = self.preprocess_factor(all_section_factor_x)
        na_free_all_section_factor_y = [self.fill_na(item) for item in all_section_factor_y]
        weights = []
        IC = []
        for i in range(len(preprocessed_all_section_factor_x)):
            x = np.array(preprocessed_all_section_factor_x[i]).reshape(-1, 1)
            y = np.array(na_free_all_section_factor_y[i])
            clf = LinearRegression()
            clf.fit(x, y)
            # print('coef:', type(clf.coef_[0]))
            # print('intercept:', clf.intercept_)
            weights.append(clf.coef_[0])
            ic = pd.concat([preprocessed_all_section_factor_x[i], na_free_all_section_factor_y[i]], axis=1).corr(method='spearman').iloc[0, 1]
            IC.append(ic)
        IC = pd.Series(IC)
        IR = IC.mean()/IC.std()
        return weights, IC, IR


if __name__ == '__main__':
    test_group = Group('720000')
    p_f = test_group.preprocess_factor(factor)
    group_return_ts = test_group.plot_group_cumsum_ret(ts=p_f)
    group_extra_performance_ts = test_group.plot_group_extra_performance(ts=p_f)
    # test_regression = Regression('720000')
    # Weights, IC, IR = test_regression.get_factor_load()