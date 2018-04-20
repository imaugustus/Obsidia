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
from statsmodels.tsa.arima_model import ARMA


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


# 计算历史因子收益率向量
def get_all_factor_ret_vector(factor_ts):
    group_stock = list(factor_ts.columns)
    factor_ret_vector = pd.Series(index=factor_ts.index[0:-1])
    bias_vector = pd.Series(index=factor_ts.index[0:-1])
    for date in factor_ts.index[0:-1]:
        today_index = factor_ts.index.get_loc(date)
        tommorrow_index = today_index + 1
        tommorrow_ret = MktData.loc[MktData.index[tommorrow_index], (group_stock, 'ret')]
        tommorrow_ret.index = tommorrow_ret.index.droplevel(level=1)
        today_factor_exposure = factor_ts.loc[date, :]
        today_factor_exposure = today_factor_exposure.fillna(0)
        tomorrow_ret = MktData.loc[MktData.index[tommorrow_index], (group_stock, 'ret')]
        tomorrow_ret = tomorrow_ret.fillna(0)
        tomorrow_ret.index = tomorrow_ret.index.droplevel(level=1)
        try:
            clf = LinearRegression()
            clf.fit(np.array(today_factor_exposure).reshape(-1, 1), tomorrow_ret)
            weights = clf.coef_[0]
            intercept = clf.intercept_
            factor_ret_vector[date] = weights
            bias_vector[date] = intercept
        except ValueError:
            print(today_factor_exposure.isna())
            print(tomorrow_ret.isna())
    return factor_ret_vector, bias_vector


# 预测某一日的因子收益率
def predict_certain_day_factor_ret(vector, date, referrence_count):
    index = vector.index.get_loc(date)
    try:
        factor_ret = vector.iloc[index-referrence_count:index].mean()
    except IndexError:
        print('Requested count of history regression parameter is out of the range of existing data, using all existing data instead')
        factor_ret = vector.iloc[:index].mean()
    return factor_ret


# 统计信息
def stats_info(vector):
    vector_mean = vector.mean()
    t = vector/vector.std()
    t_abs_mean = np.abs(t).mean()
    t_mean_abs_std = np.abs(t.mean())/t.std()
    ratio = (np.abs(t) > 2).sum()/len(vector)# IR>2
    return vector_mean, t, t_abs_mean, t_mean_abs_std, ratio


# IC and IR
def ic_ir(factor_ts):
    group_stock = list(factor_ts.columns)
    IC_ts = pd.Series(index=factor_ts.index[:-1])
    for date in factor_ts.index[:-1]:
        today_index = factor_ts.index.get_loc(date)
        tomorrow_index = today_index + 1
        tomorrow_ret = MktData.loc[MktData.index[tomorrow_index], (group_stock, 'ret')]
        tomorrow_ret.index = tomorrow_ret.index.droplevel(level=1)
        today_factor_exposure = factor_ts.loc[date, :]
        today_factor_exposure = today_factor_exposure.fillna(0)
        tomorrow_ret = MktData.loc[MktData.index[tomorrow_index], (group_stock, 'ret')]
        tomorrow_ret = tomorrow_ret.fillna(0)
        tomorrow_ret.index = tomorrow_ret.index.droplevel(level=1)
        corrcoef = np.corrcoef(today_factor_exposure, tomorrow_ret)[0, 1]
        IC_ts[date] = corrcoef
    return IC_ts, IC_ts.mean()/IC_ts.std()


if __name__ == '__main__':
    intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
    MktData = intern['MktData']
    MktData = MktData.swaplevel(0, 1, axis=1)
    industry_code = '720000'
    factor_real = pickle.load(open(r'D:/sync/Factor/v5/delta_factor_{}.pkl'.format(industry_code), 'rb'))
    factor_real = factor_real.dropna(axis=1, how='all')
    test_factor_ret_vector, test_bias_vector = get_all_factor_ret_vector(factor_real)
    all_test_date = list(MktData.index[-150:-100])
    IC_all = {}
    all_stats_info = pd.DataFrame(index=MktData.index[-150:-100], columns=['vector_mean', 't_abs_mean', 't_mean_abs_std', 'ratio', 'ir'],dtype='float')
    for date in all_test_date:
        predict_factor_ret = predict_certain_day_factor_ret(test_factor_ret_vector, date, 20)
        index = test_factor_ret_vector.index.get_loc(date)
        start = index - 20
        test_vector_mean, test_t, test_t_abs_mean, test_t_mean_abs_std, test_ratio = stats_info(test_factor_ret_vector.iloc[start:index])
        test_ic, test_ir = ic_ir(factor_real.iloc[start:index])
        IC_all[date] = test_ic
        all_stats_info.loc[date, 'vector_mean'] = test_vector_mean
        all_stats_info.loc[date, 't_abs_mean'] = test_t_abs_mean
        all_stats_info.loc[date, 't_mean_abs_std'] = test_t_mean_abs_std
        all_stats_info.loc[date, 'ratio'] = test_ratio
        all_stats_info.loc[date, 'ir'] = test_ir
        test_ic.cumsum().plot()
        plt.title(date)
        plt.show()



