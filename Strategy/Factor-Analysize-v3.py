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

    # 平均法预测因子收益
    def get_factor_ret_predict_average(self, factor_ts, date, count):
        descendant = list(factor_ts.columns)
        today_factor_exposure = factor_ts.loc[date, :]
        today_index = MktData.index.get_loc(date)
        tommorow_index = today_index + 1
        tommorow_ret = MktData.loc[MktData.index[tommorow_index], (descendant, 'ret')]
        tommorow_ret.index = tommorow_ret.index.droplevel(1)
        factor_ret_vector, bias_vector = self.get_factor_ret_vector(factor_ts, date, count=count)
        predict_factor_ret = factor_ret_vector.mean()
        predict_bias = bias_vector.mean()
        predict_tommorow_ret = predict_factor_ret*today_factor_exposure + predict_bias
        loss = np.sum((predict_tommorow_ret-tommorow_ret)**2)
        direction_precision = (predict_tommorow_ret*tommorow_ret > 0).sum()/len(tommorow_ret)
        print(loss, direction_precision)
        return predict_tommorow_ret

# 时间序列预测因子收益
    def get_factor_ret_predict_arma(self, factor_ts, date, p, count):
        descendant = list(factor_ts.columns)
        today_factor_exposure = factor_ts.loc[date, :]
        today_index = MktData.index.get_loc(date)
        tommorow_index = today_index + 1
        tommorow_ret = MktData.loc[MktData.index[tommorow_index], (descendant, 'ret')]
        tommorow_ret.index = tommorow_ret.index.droplevel(1)
        factor_ret_vector, bias_vector = self.get_factor_ret_vector(factor_ts, date, count=count)
        arma_weights = ARMA(factor_ret_vector, order=(p, 0)).fit(disp=1)
        weights_predict_arma = np.asscalar(arma_weights.forecast(1)[0])
        arma_bias = ARMA(bias_vector, order=(p, 0)).fit(disp=1)
        bias_predict_arma = np.asscalar(arma_bias.forecast(1)[0])
        predict_tommorow_ret = weights_predict_arma*today_factor_exposure+bias_predict_arma
        loss = np.sum((predict_tommorow_ret - tommorow_ret) ** 2)
        direction_precision = (predict_tommorow_ret * tommorow_ret > 0).sum() / len(tommorow_ret)
        print(loss, direction_precision)
        return predict_tommorow_ret


# 计算历史因子收益率向量
    def get_factor_ret_vector(self, factor_ts, date, count):
        group_stock = list(factor_ts.columns)
        today_index = factor_ts.index.get_loc(date)
        tommorrow_index = today_index + 1
        tommorrow_ret = MktData.loc[MktData.index[tommorrow_index], (group_stock, 'ret')]
        tommorrow_ret.index = tommorrow_ret.index.droplevel(level=1)
        weights_ts = []
        intercept_ts = []
        for i in range(today_index, today_index-count-1, -1):
            assert MktData.index[i] == factor_ts.index[i]
            yesterday_factor_exposure = factor_ts.loc[factor_ts.index[i - 1], :]
            yesterday_factor_exposure = yesterday_factor_exposure.fillna(0)
            today_ret = MktData.loc[MktData.index[i], (group_stock, 'ret')]
            today_ret = today_ret.fillna(0)
            today_ret.index = today_ret.index.droplevel(level=1)
            try:
                clf = LinearRegression()
                clf.fit(np.array(yesterday_factor_exposure).reshape(-1, 1), today_ret)
                weights = clf.coef_[0]
                intercept = clf.intercept_
                weights_ts.append(weights)
                intercept_ts.append(intercept)
            except ValueError:
                print(yesterday_factor_exposure.isna())
                print(today_ret.isna())
        factor_ret_vector = pd.Series(weights_ts, index=factor_ts.index[range(today_index-1, today_index-count-2, -1)])
        bias_vector = pd.Series(intercept_ts, index=factor_ts.index[range(today_index-1, today_index-count-2, -1)])
        return factor_ret_vector, bias_vector

    def summary_average_estiamte(self, maximum_load_count, date, count):
        result_df = pd.DataFrame(index=range(1, maximum_load_count),
                                 columns=['weights_predict', 'bias_predict', 'loss', 'direction_precision'], dtype='float')
        full_factor_ret_vector, full_bias_vector = self.get_factor_ret_vector(factor_ts=factor_real,
                                                                                         date=date, count=count)
        descendant = list(factor_real.columns)
        today_factor_exposure = factor_real.loc[date, :]
        today_index = MktData.index.get_loc(date)
        tommorow_index = today_index + 1
        tommorow_ret = MktData.loc[MktData.index[tommorow_index], (descendant, 'ret')]
        tommorow_ret.index = tommorow_ret.index.droplevel(1)
        IC = pd.DataFrame({"factor exposure":today_factor_exposure, "tommorow ret":tommorow_ret}).corr(method='spearman').iloc[0,1]
        for count in range(1, maximum_load_count):
            factor_ret_vector = full_factor_ret_vector[0:count]
            bias_vector = full_bias_vector[0:count]
            predict_factor_ret = factor_ret_vector.mean()
            predict_bias = bias_vector.mean()
            result_df.loc[count, 'weights_predict'] = predict_factor_ret
            result_df.loc[count, 'bias_predict'] = predict_bias
            predict_tommorow_ret = predict_factor_ret * today_factor_exposure + predict_bias
            loss = np.sum((predict_tommorow_ret - tommorow_ret) ** 2)/len(tommorow_ret)
            direction_precision = (predict_tommorow_ret * tommorow_ret > 0).sum() / len(tommorow_ret)
            result_df.loc[count, 'loss'] = loss
            result_df.loc[count, 'direction_precision'] = direction_precision
        return result_df, IC

    def summary_arma_estimate(self, maximum_p, date, count):
        factor_ret_vector, bias_vector = test_regression.get_factor_ret_vector(factor_real, date=date, count=count)
        result_df_arma = pd.DataFrame(index=range(1, maximum_p),
                                      columns=['weights_predict', 'bias_predict', 'loss', 'direction_precision'], dtype='float')
        descendant = list(factor_real.columns)
        today_factor_exposure = factor_real.loc[date, :]
        today_index = MktData.index.get_loc(date)
        tommorow_index = today_index + 1
        tommorow_ret = MktData.loc[MktData.index[tommorow_index], (descendant, 'ret')]
        tommorow_ret.index = tommorow_ret.index.droplevel(1)
        for p in range(1, maximum_p):
            arma_weights = ARMA(factor_ret_vector, order=(p, 0)).fit(disp=1)
            weights_predict_arma = np.asscalar(arma_weights.forecast(1)[0])
            result_df_arma.loc[p, 'weights_predict'] = weights_predict_arma
            arma_bias = ARMA(bias_vector, order=(p, 0)).fit(disp=-1)
            bias_predict_arma = np.asscalar(arma_bias.forecast(1)[0])
            result_df_arma.loc[p, 'bias_predict'] = bias_predict_arma
            predict_tommorow_ret = weights_predict_arma * today_factor_exposure + bias_predict_arma
            loss = np.sum((predict_tommorow_ret - tommorow_ret) ** 2)
            direction_precision = (predict_tommorow_ret * tommorow_ret > 0).sum() / len(tommorow_ret)
            result_df_arma.loc[p, 'loss'] = loss
            result_df_arma.loc[p, 'direction_precision'] = direction_precision
        return result_df_arma


if __name__ == '__main__':
    intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
    MktData = intern['MktData']
    MktData = MktData.swaplevel(0, 1, axis=1)
    test_maximum_load_count = 600
    test_maximum_p = 10
    test_all_count = 700
    industry_code = '720000'
    all_test_date = list(MktData.index[-150:-100])
    all_date_stats_info = {}
    all_date_stats_info_arma = {}
    IC = pd.Series(MktData.index[-150:-100])
    factor_real = pickle.load(open(r'D:/sync/Factor/v5/delta_factor_{}.pkl'.format(industry_code), 'rb'))
    largest_info = pd.DataFrame(index=MktData.index[-150:-100], columns=pd.MultiIndex.from_product([['weights_predict', 'bias_predict', 'loss', 'direction_precision'], list(range(1, 2))]))
    largest_info_arma = pd.DataFrame(index=MktData.index[-150:-100], columns=pd.MultiIndex.from_product([['weights_predict', 'bias_predict', 'loss', 'direction_precision'], list(range(1, 2))]))
    factor_real = factor_real.dropna(axis=1, how='all')
    for test_date in all_test_date:
        test_regression = Regression(industry_code)
        print("-----Analysizing stock of {} on {}-----".format(industry_code, test_date))
        df_average, ic = test_regression.summary_average_estiamte(test_maximum_load_count, date=test_date, count=test_all_count)
        for i in df_average.columns:
            if i== 'loss':
                largest_info.loc[test_date, (i, slice(None))] = df_average[i].idxmin()
            else:
                largest_info.loc[test_date, (i, slice(None))] = df_average[i].idxmax()
        IC[test_date] = ic
        all_date_stats_info[test_date] = df_average
        df_arma = test_regression.summary_arma_estimate(maximum_p=test_maximum_p, date=test_date, count=200)
        for i in df_arma.columns:
            if i=='loss':
                largest_info.loc[test_date, (i, slice(None))] = df_arma[i].idxmin()
            else:
                largest_info.loc[test_date, (i, slice(None))] = df_arma[i].idxmax()
        all_date_stats_info_arma[test_date] = df_arma

    avg_distr_precision = [all_date_stats_info[key]['direction_precision'].idxmax() for key in
                           all_date_stats_info.keys()]
    plt.hist(avg_distr_precision, bins=20)
    plt.title("Distribution of average days by precision")
    plt.xlabel("Days")
    plt.ylabel('Number')
    plt.show()
    avg_distr_loss = [all_date_stats_info[key]['loss'].idxmin() for key in all_date_stats_info.keys()]
    plt.hist(avg_distr_loss, bins=20)
    plt.title("Distribution of average days by loss")
    plt.xlabel("Days")
    plt.ylabel('Number')
    plt.show()

    arma_distr_precision = [all_date_stats_info_arma[key]['direction_precision'].idxmax() for key in
                            all_date_stats_info_arma.keys()]
    plt.hist(arma_distr_precision, bins=20)
    plt.title("Distribution of arma days by precsision")
    plt.xlabel("Days")
    plt.ylabel('Number')
    plt.show()

    arma_distr_loss = [all_date_stats_info_arma[key]['loss'].idxmin() for key in all_date_stats_info_arma.keys()]
    plt.hist(arma_distr_loss, bins=20)
    plt.title("Distribution of arma days by loss")
    plt.xlabel("Days")
    plt.ylabel('Number')
    plt.show()
