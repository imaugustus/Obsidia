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

intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
MktData = MktData.swaplevel(0, 1, axis=1)
industry_code = '720000'
factor = pickle.load(open(r'D:/sync/Factor/v4/factor_predict_ts_{}.pkl'.format(industry_code), 'rb'))
factor = factor.dropna(axis=1, how='all')
factor_real = pickle.load(open(r'D:/sync/Factor/v4/factor_train_ts_{}.pkl'.format(industry_code), 'rb'))
factor_real = factor_real.dropna(axis=1, how='all')


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
        plt.title("Cumsum Ret of Each Group Classified By factor")
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
        plt.title("Extra Peformance With Regard To Industry Average")
        plt.show()
        return group_extra_performance_ts.cumsum()

    # 检查因子的有效性
    def check_factor_efficiency(self, ts):
        mul_mean_ts = pd.DataFrame(index=ts.index, columns=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'])
        for date in ts.index:
            gb = ts.loc[date, :].groupby(pd.cut(ts.loc[date, :], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
            for group_label in gb.groups.keys():
                group_stock = list(gb.groups[group_label])
                today_ret = MktData.loc[date, (group_stock, 'ret')]
                today_ret.index = today_ret.index.droplevel(level=1)
                today_index = MktData.index.get_loc(date)
                tommor_index = today_index + 1
                tomorrow_ret = MktData.loc[MktData.index[tommor_index], (group_stock, 'ret')]
                tomorrow_ret.index = tomorrow_ret.index.droplevel(level=1)
                delta = tomorrow_ret - today_ret
                f = ts.loc[date, group_stock]
                mul = f.multiply(delta)
                mul_sum = mul.sum()
                mul_mean = mul.mean()
                mul_mean_ts.loc[date, group_label] = mul_mean
        mul_mean_ts.cumsum().plot()
        plt.show()
        return mul_mean_ts.cumsum()

    # 检查预测因子的有效性
    def check_factor_predict_efficiency(self, ts):
        mul_mean_predict_ts = pd.DataFrame(index=ts.index, columns=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'])
        for date in ts.index:
            gb = ts.loc[date, :].groupby(pd.cut(ts.loc[date, :], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
            for group_label in gb.groups.keys():
                group_stock = list(gb.groups[group_label])
                today_ret = MktData.loc[date, (group_stock, 'ret')]
                today_ret.index = today_ret.index.droplevel(level=1)
                today_index = MktData.index.get_loc(date)
                tommor_index = today_index + 1
                tomorrow_ret = MktData.loc[MktData.index[tommor_index], (group_stock, 'ret')]
                tomorrow_ret.index = tomorrow_ret.index.droplevel(level=1)
                delta = tomorrow_ret - today_ret
                f = ts.loc[date, group_stock]
                mul = f.multiply(delta)
                mul_sum = mul.sum()
                mul_mean = mul.mean()
                mul_mean_predict_ts.loc[date, group_label] = mul_mean
        mul_mean_predict_ts.cumsum().plot()
        plt.show()
        return mul_mean_predict_ts.cumsum()


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

# 平均法预测因子收益
    def multi_regression_average(self, factor_ts, Mkt_ts, date, param_df, previous_load_count):
        gb = factor_ts.loc[date, :].groupby(pd.cut(factor_ts.loc[date, :], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
        today_factor_index = factor_ts.index.get_loc(date)
        today_ret_index = Mkt_ts.index.get_loc(date)
        tommorrow_ret_index = today_ret_index + 1
        for key in gb.groups.keys():
            group_stock = list(gb.groups[key])
            today_factor_exposure = factor_ts.loc[date, group_stock]
            tommorrow_ret = Mkt_ts.loc[Mkt_ts.index[tommorrow_ret_index], (group_stock, 'ret')]
            tommorrow_ret.index = tommorrow_ret.index.droplevel(level=1)
            weights = 0
            intercept = 0
            for factor_index, mkt_index in zip(range(today_factor_index, today_factor_index-previous_load_count-1, -1), range(today_ret_index, today_ret_index-previous_load_count-1, -1)):
                assert Mkt_ts.index[mkt_index] == factor_ts.index[factor_index]
                assert Mkt_ts.index[mkt_index+1] == factor_ts.index[factor_index+1]
                temp_factor_exposure = factor_ts.loc[factor_ts.index[factor_index - 1], group_stock]
                temp_factor_exposure = temp_factor_exposure.fillna(0)
                temp_mkt_ret = Mkt_ts.loc[Mkt_ts.index[mkt_index], (group_stock, 'ret')]
                temp_mkt_ret = temp_mkt_ret.fillna(0)
                temp_mkt_ret.index = temp_mkt_ret.index.droplevel(level=1)
                try:
                    clf = LinearRegression()
                    clf.fit(np.array(temp_factor_exposure).reshape(-1, 1), temp_mkt_ret)
                    weights += clf.coef_[0]
                    intercept += clf.intercept_
                except ValueError:
                    print(temp_factor_exposure.isna())
                    print(temp_mkt_ret.isna())
            weights /= previous_load_count
            intercept /= previous_load_count
            param_df.loc[previous_load_count, (key, 'weights')] = weights
            param_df.loc[previous_load_count, (key, 'intercept')] = intercept
            predict_tommorrow_ret = weights*today_factor_exposure + intercept
            mul = tommorrow_ret*predict_tommorrow_ret
            directioin_precision = (mul > 0).sum()/len(mul)
            print(key, directioin_precision)


# 时间序列预测因子收益
    def multi_regression_arma(self, factor_ts, Mkt_ts, date, param_df, previous_load_count=300):
        gb = factor_ts.loc[date, :].groupby(pd.cut(factor_ts.loc[date, :], bins=5, labels=['G-5', 'G-4', 'G-3', 'G-2', 'G-1'], retbins=False))
        today_factor_index = factor_ts.index.get_loc(date)
        today_ret_index = Mkt_ts.index.get_loc(date)
        tommorrow_ret_index = today_ret_index + 1
        for key in gb.groups.keys():
            group_stock = list(gb.groups[key])
            today_factor_exposure = factor_ts.loc[date, group_stock]
            tommorrow_ret = Mkt_ts.loc[Mkt_ts.index[tommorrow_ret_index], (group_stock, 'ret')]
            tommorrow_ret.index = tommorrow_ret.index.droplevel(level=1)
            weights_ts = []
            intercept_ts = []
            for factor_index, mkt_index in zip(range(today_factor_index, today_factor_index-previous_load_count-1, -1), range(today_ret_index, today_ret_index-previous_load_count-1, -1)):
                assert Mkt_ts.index[mkt_index] == factor_ts.index[factor_index]
                assert Mkt_ts.index[mkt_index+1] == factor_ts.index[factor_index+1]
                temp_factor_exposure = factor_ts.loc[factor_ts.index[factor_index - 1], group_stock]
                temp_factor_exposure = temp_factor_exposure.fillna(0)
                temp_mkt_ret = Mkt_ts.loc[Mkt_ts.index[mkt_index], (group_stock, 'ret')]
                temp_mkt_ret = temp_mkt_ret.fillna(0)
                temp_mkt_ret.index = temp_mkt_ret.index.droplevel(level=1)
                try:
                    clf = LinearRegression()
                    clf.fit(np.array(temp_factor_exposure).reshape(-1, 1), temp_mkt_ret)
                    weights = clf.coef_[0]
                    intercept = clf.intercept_
                    weights_ts.append(weights)
                    intercept_ts.append(intercept)
                except ValueError:
                    print(temp_factor_exposure.isna())
                    print(temp_mkt_ret.isna())
            for p in range(1, 30):
                arma_weights = ARMA(weights_ts, order=(p, 0)).fit(disp=-1)
                predict_factor_ret = np.asscalar(arma_weights.forecast(1)[0])
                arma_intercept = ARMA(intercept_ts, order=(p, 0)).fit(disp=-1)
                predict_factor_intercept = np.asscalar(arma_intercept.forecast(1)[0])
                param_df.loc[p, (key, 'weights')] = predict_factor_ret
                param_df.loc[p, (key, 'intercept')] = predict_factor_intercept
                predict_ret = predict_factor_ret*today_factor_exposure+predict_factor_intercept
                mul = tommorrow_ret*predict_ret
                precision = (mul > 0).sum()/len(mul)
                print(key, precision)


if __name__ == '__main__':
    # test_group = Group(industry_code)
    # p_f = test_group.preprocess_factor(factor)
    # p_f_r = test_group.preprocess_factor(factor_real)
    # group_return_ts = test_group.plot_group_cumsum_ret(ts=p_f_r)
    # group_return_ts_p = test_group.plot_group_cumsum_ret(ts=p_f)
    # group_extra_performance_ts = test_group.plot_group_extra_performance(ts=p_f_r)
    # group_extra_performance_ts_p = test_group.plot_group_extra_performance(ts=p_f)
    # mul_mena_ts = test_group.check_factor_efficiency(p_f_r)
    # mul_mena_predict_ts = test_group.check_factor_efficiency(p_f)
    test_regression = Regression('720000')
    # p_f = test_regression.preprocess_factor(factor_real)
    # average_param_df = pd.DataFrame(index=range(1, 150),
    #             columns=pd.MultiIndex.from_product([['G-5', 'G-4', 'G-3', 'G-2', 'G-1'],['weights', 'intercept']]))
    arma_param_df = pd.DataFrame(index=range(1, 30),
                columns=pd.MultiIndex.from_product([['G-5', 'G-4', 'G-3', 'G-2', 'G-1'],['weights', 'intercept']]))
    # for i in range(1, 150):
    #     test_regression.multi_regression_average(factor_real, MktData, '2018-01-29', param_df=average_param_df, previous_load_count=i)
    test_regression.multi_regression_arma(factor_real, MktData, '2018-01-29', param_df=arma_param_df,
                                          previous_load_count=300)
    # average_param_df.plot()
    # plt.show()
    arma_param_df.plot()
    plt.show()