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
    factor_section = factor_section.fillna(factor_section)
    return factor_section


# 预处理因子
def preprocess_factor(factor):
    preprocessed_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    for date in factor.index:
        section_factor = factor.loc[date, :]
        section_factor = filter_extreme(section_factor)
        section_factor = normalize(section_factor)
        section_factor = fill_na(section_factor)
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
    summary = pd.DataFrame(index=relative_ret.index, columns=['factor-ret of lag {}'.format(lag), 'R-square'])
    for i in range(lag+1, len(relative_ret.index)):
        y = relative_ret.iloc[i, :].astype(float)
        x = preprocessed_factor.iloc[i-lag-1, :].astype(float)
        X = sm.add_constant(x)
        model = sm.OLS(y, X, missing='drop')
        results = model.fit()
        fitted_values = results.fittedvalues
        weights = results.params[1]
        summary.iloc[i, 0] = weights
        summary.iloc[i, 1] = results.rsquared
        # new_x = x.drop(x.index[model.data.missing_row_idx])
        # if i % 100 == 0:
        #     plt.plot(new_x, fitted_values, 'r--', label='OLS')
        #     plt.plot(x, y, 'o', label='data')
        #     plt.legend(loc='best')
        #     plt.show()
        #     print(results.summary())
    return summary


def lag_correlation(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    summary = pd.Series(index=relative_ret.index)
    for i in range(lag+1, len(relative_ret.index)):
        y = relative_ret.iloc[i, :].astype(float)
        x = preprocessed_factor.iloc[i-lag-1, :].astype(float)
        corr = pd.concat([x, y], axis=1).corr(method='spearman').iloc[0, 1]
        summary[i] = corr
    return summary


def lag_group_ret(lag, relative_ret):
    preprocessed_factor = preprocess_factor(relative_ret)
    summary_relative = pd.DataFrame(index=relative_ret.index, columns=['Extreme Low', 'Low', 'Medium', 'High', 'Extreme High'])
    summary_abs = pd.DataFrame(index=relative_ret.index, columns=['Extreme Low', 'Low', 'Medium', 'High', 'Extreme High'])
    for i in range(lag+1, len(relative_ret.index)):
        gb = relative_ret.iloc[i, :].groupby(pd.cut(preprocessed_factor.iloc[i-lag-1, :], bins=5, labels=['Extreme Low', 'Low', 'Medium', 'High', 'Extreme High'], retbins=False))
        for key in gb.groups.keys():
            # print(key)
            group = list(gb.groups[key])
            summary_relative.loc[relative_ret.index[i], key] = relative_ret.loc[relative_ret.index[i], group].mean()
            summary_abs.loc[relative_ret.index[i], key] = MktData.loc[relative_ret.index[i], (group, 'ret')].mean()
    return summary_relative, summary_abs


if __name__ == '__main__':
    ret_ = data_input()
    all_factor_ret = []
    all_corr = []
    all_ret_relative = []
    all_ret_abs = []
    for lag in range(8):
        sum_regression = lag_regression(lag, ret_)
        all_factor_ret.append(sum_regression['factor-ret of lag {}'.format(lag)])
        sum_corr = lag_correlation(lag, ret_)
        all_corr.append(sum_corr)
        sum_relative, sum_abs = lag_group_ret(lag, ret_)
        sum_relative.cumsum(axis=0).plot()
        plt.title("Cumsum of relative ret of lag {}".format(lag))
        plt.xlabel("Date")
        plt.ylabel("Cumsum ret")
        plt.show()
        sum_abs.cumsum(axis=0).plot()
        plt.title("Cumsum of abs ret of lag {}".format(lag))
        plt.xlabel("Date")
        plt.ylabel("Cumsum ret")
        plt.show()
        all_ret_relative.append(sum_relative)
        all_ret_abs.append(sum_abs)
    df_regression = pd.concat(all_factor_ret, axis=1)
    df_regression.cumsum(axis=0).plot()
    plt.title("Cumsum of factor ret")
    plt.xlabel("Date")
    plt.ylabel("Factor ret")
    plt.show()
    df_corr = pd.concat(all_corr, axis=1)
    df_corr.cumsum(axis=0).plot()
    plt.title("Cumsum of corr")
    plt.xlabel("Date")
    plt.ylabel("Corr")
    plt.show()