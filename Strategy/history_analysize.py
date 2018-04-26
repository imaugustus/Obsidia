import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import re
import math
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from scipy.stats import spearmanr

intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
InstrumentInfo = intern['InstrumentInfo']
MktData = MktData.swaplevel(0, 1, axis=1)

# 中位数去极值
def filter_extreme(factor_section, n=5):
    Dm = factor_section.median()
    Dm1 = ((factor_section-Dm).abs()).median()
    max_limit = Dm + n*Dm1
    min_limit = Dm - n*Dm1
    factor_section = np.clip(factor_section, min_limit, max_limit)
    return factor_section


# 标准化
def normalize(factor_section):
    mean = factor_section.mean()
    std = factor_section.std()
    factor_section = (factor_section - mean)/std
    return factor_section


# 缺失值处理
def fill_na(factor_section):
    factor_section = factor_section.fillna(factor_section.mean())
    return factor_section


# 预处理因子
def preprocess_factor(factor):
    preprocessed_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    for date in factor.index:
        section_factor = factor.loc[date, :]
        section_factor = filter_extreme(section_factor)
        # section_factor = normalize(section_factor)
        # section_factor = fill_na(section_factor)
        preprocessed_factor.loc[date, :] = section_factor
    return preprocessed_factor


def get_industry_descendant(industry_code, descendant_order=0):
    pattern = re.compile(industry_code[0:3+descendant_order])
    descendant = []
    for stock_code, representation in zip(InstrumentInfo.index, InstrumentInfo['SWICS']):
        if re.match(pattern, representation):
            descendant.append(stock_code)
    return descendant


def data_input(industry_code='430000'):
    descendant = get_industry_descendant(industry_code, descendant_order=0)
    ret = MktData.loc[:, (descendant, 'ret')]
    ret.columns = ret.columns.droplevel(level=1)
    industry_mean = ret.mean(axis=1)
    # TODO Target ret should be stock ret - industry_mean but then it becomes same as factor exposure
    relative_ret = ret.subtract(industry_mean, axis=0)
    return relative_ret


def lag_regression(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    factor_ret = pd.Series(index=relative_ret.index)
    rsquare = pd.Series(index=relative_ret.index)
    first_trading_day = InstrumentInfo.loc[list(relative_ret.columns), 'FirstTradingDate']
    for i in range(lag+1, len(relative_ret.index)):
        delta = relative_ret.index[i] - first_trading_day
        not_new_stock = list(delta[delta > pd.to_timedelta(120, unit='day')].index)
        # print("{} new stock has been dropped from industry".format(len(relative_ret.columns)-len(not_new_stock)))
        y = relative_ret.loc[relative_ret.index[i], not_new_stock].astype(float)
        x = preprocessed_factor.loc[relative_ret.index[i-lag-1], not_new_stock].astype(float)
        X = sm.add_constant(x)
        model = sm.OLS(y, X, missing='drop')
        results = model.fit()
        # fitted_values = results.fittedvalues
        weights = results.params[1]
        factor_ret[i] = weights
        rsquare[i] = results.rsquared
        # new_x = x.drop(x.index[model.data.missing_row_idx])
        # if i % 100 == 0:
        #     plt.plot(new_x, fitted_values, 'r--', label='OLS')
        #     plt.plot(x, y, 'o', label='data')
        #     plt.legend(loc='best')
        #     plt.show()
        #     print(results.summary())
    return factor_ret, rsquare


def lag_correlation(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    summary = pd.Series(index=relative_ret.index)
    first_trading_day = InstrumentInfo.loc[list(relative_ret.columns), 'FirstTradingDate']
    for i in range(lag+1, len(relative_ret.index)):
        delta = relative_ret.index[i] - first_trading_day
        not_new_stock = list(delta[delta > pd.to_timedelta(120, unit='day')].index)
        y = relative_ret.loc[relative_ret.index[i], not_new_stock].astype(float)
        x = preprocessed_factor.loc[relative_ret.index[i-lag-1], not_new_stock].astype(float)
        corr = pd.concat([x, y], axis=1).corr(method='spearman').iloc[0, 1]
        summary[i] = corr
    return summary


# def lag_group_ret(lag, relative_ret):
#     preprocessed_factor = preprocess_factor(relative_ret)
#     rng = np.arange(5)
#     summary_relative = pd.DataFrame(index=relative_ret.index, columns=rng)
#     summary_abs = pd.DataFrame(index=relative_ret.index, columns=rng)
#     first_trading_day = InstrumentInfo.loc[list(relative_ret.columns), 'FirstTradingDate']# 所有股票的上市日期
#     for i in range(lag+1, len(relative_ret.index)):
#         delta = relative_ret.index[i] - first_trading_day # 今日与上市日期的差值
#         not_new_stock = list(delta[delta > pd.to_timedelta(120, unit='day')].index)# 上述差值>120的定义为非新股票
#         gb = relative_ret.loc[relative_ret.index[i], not_new_stock].groupby(pd.cut(preprocessed_factor.loc[preprocessed_factor.index[i-lag-1], not_new_stock], bins=5, labels=rng, retbins=False))
#         for key in gb.groups.keys():
#             group = list(gb.groups[key])
#             summary_relative.loc[relative_ret.index[i], key] = relative_ret.loc[relative_ret.index[i], group].mean()
#             summary_abs.loc[relative_ret.index[i], key] = MktData.loc[relative_ret.index[i], (group, 'ret')].mean()
#     return summary_relative, summary_abs


def lag_group_ret(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    rng = list(range(5))
    summary_relative = pd.DataFrame(index=relative_ret.index, columns=rng)
    summary_abs = pd.DataFrame(index=relative_ret.index, columns=rng)
    first_trading_day = InstrumentInfo.loc[list(relative_ret.columns), 'FirstTradingDate']# 所有股票的上市日期
    for i in range(lag+1, len(relative_ret.index)):
        delta = relative_ret.index[i] - first_trading_day # 今日与上市日期的差值
        not_new_stock = list(delta[delta > pd.to_timedelta(120, unit='day')].index)# 上述差值>120的定义为非新股票
        gb = relative_ret.loc[relative_ret.index[i], not_new_stock].groupby(pd.qcut(preprocessed_factor.loc[preprocessed_factor.index[i-lag-1], not_new_stock], q=5, retbins=False, duplicates='drop'))
        for index, key in enumerate(gb.groups.keys()):
            group = list(gb.groups[key])
            summary_relative.loc[relative_ret.index[i], index] = relative_ret.loc[relative_ret.index[i], group].mean()
            summary_abs.loc[relative_ret.index[i], index] = MktData.loc[relative_ret.index[i], (group, 'ret')].mean()
    return summary_relative, summary_abs


# 增加换手率计算部分
def portfolio_turnover_ratio(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    rng = np.arange(5)
    first_trading_day = InstrumentInfo.loc[list(relative_ret.columns), 'FirstTradingDate']
    gb_all = {}
    turnover_ratio = pd.DataFrame(index=relative_ret.index[lag+2:], columns=rng)
    for i in range(lag+1, len(relative_ret.index)):# 计算所有日期的分组情况
        delta = relative_ret.index[i] - first_trading_day
        not_new_stock = list(delta[delta > pd.to_timedelta(120, unit='day')].index)
        gb = relative_ret.loc[relative_ret.index[i], not_new_stock].groupby(
            pd.qcut(preprocessed_factor.loc[preprocessed_factor.index[i - lag - 1], not_new_stock], q=5, retbins=False, duplicates='drop'))
        gb_all[i] = gb
    for j in range(lag+2, len(relative_ret.index)):# 计算所有天数的换手率
        today_portfolio = gb_all[j]
        yesterday_portfolio = gb_all[j-1]
        for k in range(5):
            yesterday_length = len(yesterday_portfolio.groups[k])# 昨日持股数目
            today_length = len(today_portfolio.groups[k])# 今日持股数目
            yesterday_stock = list(yesterday_portfolio.groups[k])# 昨日持股代码
            today_stock = list(today_portfolio.groups[k])# 今日持股代码
            sold_stock = [item for item in yesterday_stock if item not in today_stock]# 今日卖出的股票
            ratio = 1 - (yesterday_length - len(sold_stock))/today_length
            turnover_ratio.loc[relative_ret.index[j], k] = ratio
    return turnover_ratio


if __name__ == '__main__':
    ret_ = data_input('720000')
    # a, b = lag_group_ret(0, ret_)
    c = portfolio_turnover_ratio(0, ret_)
    # # df_turnover_ratio = portfolio_turnover_ratio(0, ret_)
    # all_factor_ret = []
    # all_rsquare = []
    # all_corr = []
    # all_ret_relative = []
    # all_ret_abs = []
    # for lag in range(5):
    #     regression_factor_ret, regression_rsquare = lag_regression(lag, ret_)
    #     all_factor_ret.append(regression_factor_ret)
    #     all_rsquare.append(regression_rsquare)
    #     sum_corr = lag_correlation(lag, ret_)
    #     all_corr.append(sum_corr)
    #     sum_relative, sum_abs = lag_group_ret(lag, ret_)
    #     sum_relative.cumsum(axis=0).plot()
    #     plt.title("Cumsum of relative ret of lag {}".format(lag))
    #     plt.xlabel("Date")
    #     plt.ylabel("Cumsum ret")
    #     plt.show()
    #     sum_abs.cumsum(axis=0).plot()
    #     plt.title("Cumsum of abs ret of lag {}".format(lag))
    #     plt.xlabel("Date")
    #     plt.ylabel("Cumsum ret")
    #     plt.show()
    #     all_ret_relative.append(sum_relative)
    #     all_ret_abs.append(sum_abs)
    # df_factor_ret = pd.concat(all_factor_ret, axis=1)
    # df_factor_ret.cumsum(axis=0).plot()
    # plt.title("Cumsum of factor ret")
    # plt.xlabel("Date")
    # plt.ylabel("Factor ret")
    # plt.show()
    # df_corr = pd.concat(all_corr, axis=1)
    # df_corr.cumsum(axis=0).plot()
    # plt.title("Cumsum of corr")
    # plt.xlabel("Date")
    # plt.ylabel("Corr")
    # plt.show()
    # df_rsquare = pd.concat(all_rsquare, axis=1)
    # df_rsquare.cumsum(axis=0).plot()
    # plt.title("Cumsum of rsquare")
    # plt.xlabel("Date")
    # plt.ylabel("Rsquare")
    # plt.show()
    # # df_turnover_ratio.cumsum(axis=0).plot()
    # # plt.title("Cumsum of turnover_ratio")
    # # plt.xlabel("Date")
    # # plt.ylabel("Turnover ratio")
    # # plt.show()
