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
factor = pickle.load(open(r'D:/sync/Factor/factor_real_ts.pkl', 'rb'))
factor = factor.dropna(axis=1, how='all')

class Group:
    def __init__(self, industry_code):
        self.industry_code = industry_code
        # self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        # self.MktData = self.intern['MktData']
        # self.MktData = self.MktData.swaplevel(0, 1, axis=1)
        # self.factor = pickle.load(open(r'D:/sync/Factor/factor_real_ts.pkl', 'rb'))
        # self.factor = self.factor.dropna(axis=1, how='all')

# 按照间隔获取所有分类截面
    def get_group_section(self, period=30):
        temp = []
        for i in range(0, len(factor.index), period):
            temp.append(factor.iloc[i, :])
        return pd.concat(temp, axis=1).transpose()

# 对某个具体截面的因子进行分类并返回group:[stock]形式的dict
    def factor_classification(self, section):
        sorted_section = section.sort_values()
        group_length = round(len(sorted_section)/5)
        classification_index = {}
        for i in range(5):
            if i == 4:
                classification_index[i+1] = sorted_section.index[i*group_length:]
            else:
                classification_index[i+1] = sorted_section.index[i*group_length:(i+1)*group_length]
        return classification_index

# 对所有截面按照因子大小对一个行业的所有股票进行分层并返回section：group_dict形式的dict分层结果
    def classification_all_section(self):
        sections = self.get_group_section()
        ts_group_result = {}
        for section in sections.index:
            # key = datetime.fromtimestamp(section).strftime('%Y-%m-%dT%H:%M:%SZ')
            key = str(section)
            ts_group_result[key] = self.factor_classification(sections.loc[section, :])
        return ts_group_result

# 对每个截面按照分层结果画每个group的累计收益图
    def plot_group_cumsum_ret(self, ts_group_result):
        dates = list(ts_group_result.keys())
        all_cumsum_ret = {}
        for i, date in enumerate(ts_group_result.keys()):
            group_result = ts_group_result[date]
            hierarchy_all = []
            for group in group_result.keys():
                group_stock = list(group_result[group])
                data = MktData.loc['2016-01-01':, (group_stock, 'ret')]
                hierarchy_average = data.mean(axis=1)
                hierarchy_all.append(hierarchy_average)
            df = pd.concat(hierarchy_all, axis=1).cumsum()
            # if i%4 == 0:
            df.plot(title="Group Average Ret classified by {} Section".format(str(date)))
            plt.show()
            all_cumsum_ret[i] = df
        return all_cumsum_ret

# 对每个截面按照分层结果画每个group的超额收益图
    def plot_group_extra_performance(self, ts_group_result):
        dates = list(ts_group_result.keys())
        all_extra_performance = {}
        for i, date in enumerate(ts_group_result.keys()):
            group_result = ts_group_result[date]
            hierarchy_all = []
            for group in group_result.keys():
                group_stock = list(group_result[group])
                data = factor.loc['2016-01-01':, group_stock]
                hierarchy_average = data.mean(axis=1)
                hierarchy_all.append(hierarchy_average)
            df = pd.concat(hierarchy_all, axis=1)
            if i % 4 == 0:
                df.plot(title="Group Extra Performance Classified by {} Section ".format(str(date)))
                plt.show()
            all_extra_performance[i] = df
        return all_extra_performance


class Regression:
    def __init__(self, industry_code):
        self.industry_code = industry_code
        # self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        # self.MktData = self.intern['MktData']
        # self.MktData = self.MktData.swaplevel(0, 1, axis=1)
        # self.factor = pickle.load(open(r'D:/sync/Factor/factor_real_ts.pkl', 'rb'))

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

# 获取截面期因子暴露以及下个月平均收益
    def get_train_data(self, factor_ts):
        stock = list(factor_ts.columns)
        all_section_factor_x = []
        all_section_factor_y = []
        for section in range(0, factor_ts.shape[0], 32):
            section_factor_x = factor_ts.iloc[section, :]
            all_section_factor_x.append(section_factor_x)
            section_date = factor_ts.index[section]
            # print("Factor section date:", section_date)
            index = MktData.index.get_loc(section_date)
            ret_start_index = index + 1
            ret_end_index = index + 31
            ret = MktData.loc[MktData.index[ret_start_index:ret_end_index], (stock, 'ret')]
            section_factor_y = ret.mean(axis=0)
            section_factor_y.index = section_factor_y.index.droplevel(1)
            all_section_factor_y.append(section_factor_y)
            # ret_start_date = MktData.index[ret_start_index]
            # ret_end_date = MktData.index[ret_end_index]
            # print('ret start date:', ret_start_date)
            # print('ret end date:', ret_end_date)
        return all_section_factor_x, all_section_factor_y

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
    ts = test_group.classification_all_section()
    all_df_cumsum_ret = test_group.plot_group_cumsum_ret(ts)
    all_extra_performance = test_group.plot_group_extra_performance(ts)
    test_regression = Regression('720000')
    Weights, IC, IR = test_regression.get_factor_load()