import numpy as np
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif']=['SimHei']
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
import pickle
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import re
import numpy
from pandas import tseries



intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
InstrumentInfo = intern['InstrumentInfo']
First_Tradiung_Date = intern['InstrumentInfo']['FirstTradingDate']
code_first_MktData = MktData.swaplevel(0, 1, axis=1)
# 去除ST股票
def drop_st():
    dropped_code = []
    pattern = re.compile(r"\*ST")
    for code, name in zip(InstrumentInfo.index, InstrumentInfo['Name']):
        if re.match(pattern, name):
            dropped_code.append(code)
    no_st_MktData = code_first_MktData.drop(dropped_code, axis="columns", level=0)
    return no_st_MktData, dropped_code


no_st_code_first_MktData, dropped_code = drop_st()


class Strategy:
    def __init__(self, industry_code='720000', start='2016-01-04', time_period=30, industry_order=3):
        self.industry_code = industry_code
        self.start = start
        self.time_period = time_period
        self.industry_order = industry_order

    def get_industry_descendant(self, industry_code, industry_order=3):
        no_st_InstrumentInfo = InstrumentInfo.drop(dropped_code, axis="index")
        pattern = re.compile(industry_code[0:industry_order])
        descendant = []
        for index, stock in zip(no_st_InstrumentInfo.index, no_st_InstrumentInfo['SWICS']):
            if re.match(pattern, stock):
                descendant.append(index)
        return descendant

    def cal_valid_start_trading_day(self, industry_stock):
        valid_start_day = {}
        for stock in industry_stock:
            start_trading_day = First_Tradiung_Date[stock]
            stock_timestamp = no_st_code_first_MktData.loc[start_trading_day, stock].name
            index = no_st_code_first_MktData.index.get_loc(stock_timestamp)
            valid_start_trading_index = index+60 #以开盘日60天之后作为数据起始点
            valid_start_trading_day = no_st_code_first_MktData.ix[valid_start_trading_index, stock].name
            valid_start_day[stock] = valid_start_trading_day
        return valid_start_day


# 计算某一天开始后30天内行业指数以及各支股票相对行业指数的超额收益率
    def category_index(self, industry_code, start='2016-01-01', time_period=30, industry_order=3):
        start_index = no_st_code_first_MktData.index.get_loc(start)
        end_index = start_index+time_period
        descendant = self.get_industry_descendant(industry_code, industry_order)
        industry_all = no_st_code_first_MktData.loc[start:no_st_code_first_MktData.index[end_index], (descendant, 'ret')]
        industry_all.columns = industry_all.columns.droplevel(level=1)
        na_free_industry_all = industry_all.dropna(axis=1, how='any')
        industry_index = na_free_industry_all.mean(axis=1, skipna=True)
        extra_stock_performance = na_free_industry_all.sub(industry_index, axis=0)
        return industry_index, extra_stock_performance

# 计算对应于从某一天开始的30个标准交易日后的真实超额收益，用以和预测超额收益进行对比
    def cal_real_extra_performance(self, industry_code='720000', start='2016-01-01', time_period=30, industry_order=3):
        start_index = no_st_code_first_MktData.index.get_loc(start)
        end_index = start_index + time_period
        real_index = start_index + time_period + 1
        predict_datetimne = no_st_code_first_MktData.index[real_index]
        descendant = self.get_industry_descendant(industry_code, industry_order)
        industry_all = no_st_code_first_MktData.loc[start:no_st_code_first_MktData.index[end_index], (descendant, 'ret')]
        industry_all.columns = industry_all.columns.droplevel(level=1)
        na_free_industry_all = industry_all.dropna(axis=1, how='any')
        na_dropped_stock = industry_all.columns[~industry_all.columns.isin(na_free_industry_all.columns)]
        no_na_descendant = [item for item in descendant if item not in na_dropped_stock]
        real_ret = no_st_code_first_MktData.loc[no_st_code_first_MktData.index[real_index], (no_na_descendant, 'ret')]
        real_ret.index = real_ret.index.droplevel(level=1)
        real_extra_performace = real_ret - real_ret.mean()
        return real_extra_performace, predict_datetimne

    def arma_forecast(self, ts,  p, q):
        arma = ARMA(ts, order=(p, q)).fit(disp=-1)
        # ts_predict = arma.predict()
        next_ret = arma.forecast(1)[0]
        return next_ret, arma.summary2()

    def arima_forecast(self, ts,  p, i, q,):
        arima = ARIMA(ts, order=(p, i, q)).fit(disp=-1)
        # ts_predict = arima.predict()
        next_ret = arima.forecast(1)[0]
        return next_ret, arima.summary2()

# 利用30天的超额收益训练并预测下一天的超额收益
    def train_forecast(self):
        industry_index, extra_stock_performance = self.category_index(industry_code=self.industry_code,\
                                        start=self.start, time_period=self.time_period, industry_order=self.industry_order)
        real_extra_performance, predict_datetime = self.cal_real_extra_performance(self.industry_code, start=self.start,\
                                                    time_period=self.time_period, industry_order=self.industry_order)
        predict_factor = pd.DataFrame(index=extra_stock_performance.columns, columns=['Estimated', 'Real', 'Rightness'],dtype='float')
        strategy_summary = {}
        assert extra_stock_performance.columns.any(), real_extra_performance.index.any()
        for i, stock in enumerate(extra_stock_performance.columns):
            predict_extra_performance, stock_summary = self.arma_forecast(extra_stock_performance[stock], 1, 0)
            strategy_summary[stock] = stock_summary
            predict_factor.loc[stock, 'Estimated'] = float(predict_extra_performance)
            predict_factor.loc[stock, 'Real'] = real_extra_performance[stock]
            predict_factor.loc[stock, 'Rightness'] = 1 if float(predict_extra_performance)*real_extra_performance[stock] > 0 else -1
        return predict_factor, predict_datetime, strategy_summary


if __name__ == '__main__':
    ts_stats_info = pd.DataFrame(columns=['Square_loss_sum', 'Corr', 'Precision', 'Mean', 'Std', 'Skew', 'Kurt'], dtype='float')
    industry_code_i = '720000'
    time_period_i = 30
    industry_order_i = 3
    start = '2016-01-04'
    end = '2018-01-04'
    start_index = no_st_code_first_MktData.index.get_loc(start)
    end_index = no_st_code_first_MktData.index.get_loc(end)
    delta = 31
    predict_start_index = start_index + delta
    predict_end_index = end_index + delta
    predict_start_traing_date = no_st_code_first_MktData.index[predict_start_index]
    predict_end_traing_date = no_st_code_first_MktData.index[predict_end_index]
    rng_train = no_st_code_first_MktData.index[start_index:end_index]
    rng_predict = no_st_code_first_MktData.index[predict_start_index:predict_end_index]
    test = Strategy(industry_code_i, start, time_period_i, industry_order_i)
    test_descendant = test.get_industry_descendant(industry_code=industry_code_i, industry_order=3)
    factor_train_ts = pd.DataFrame(columns=test_descendant, index=rng_train)
    factor_predict_ts = pd.DataFrame(columns=test_descendant, index=rng_predict)
    for i, date in enumerate(rng_train):
        train_start_date = rng_train[i]
        predict_date = rng_predict[i]
        try:
            strategy = Strategy(industry_code_i, train_start_date, time_period_i, industry_order_i)
            factor, predict_datetime, summary = strategy.train_forecast()
            print("Predicting factor of {}".format(predict_date))
            assert predict_date == predict_datetime
            for stock in factor.index:
                try:
                    factor_predict_ts.loc[predict_date, stock] = factor.loc[stock, 'Estimated']
                    factor_train_ts.loc[predict_date, stock] = factor.loc[stock, 'Real']
                except KeyError:
                    continue
                except ValueError:
                    continue
            factor = factor.dropna(axis=0, how='any')
            Same_Direction_Prediction = factor.loc[factor['Rightness'] == 1]['Rightness'].sum()
            Diff_Direction_Prediction = factor.loc[factor['Rightness'] == -1]['Rightness'].sum()
            max_5_prediction = factor['Estimated'].sort_values(axis=0, ascending=False).head(5)
            factor['square_loss'] = (factor['Estimated']-factor['Real'])**2
            squre_loss_sum = factor['square_loss'].sum()/factor.shape[0]
            corr = factor[['Estimated', 'Real']].corr()
            Precision = Same_Direction_Prediction / (Same_Direction_Prediction - Diff_Direction_Prediction)# 反方向的正确度累积和为负数，因此用减法
            Mean = factor['Estimated'].mean()
            Std = factor['Estimated'].std()
            Skew = factor['Estimated'].skew()
            Kurt = factor['Estimated'].kurtosis()
            ts_stats_info.loc[predict_date, 'Square_loss_sum'] = squre_loss_sum
            ts_stats_info.loc[predict_date, 'Corr'] = corr.iloc[0, 1]
            ts_stats_info.loc[predict_date, 'Precision'] = Precision
            ts_stats_info.loc[predict_date, 'Mean'] = Mean
            ts_stats_info.loc[predict_date, 'Std'] = Std
            ts_stats_info.loc[predict_date, 'Skew'] = Skew
            ts_stats_info.loc[predict_date, 'Kurt'] = Kurt
        except KeyError:
            continue
    factor_train_ts = factor_train_ts.dropna(axis=0, how='all')
    factor_predict_ts = factor_predict_ts.dropna(axis=0, how='all')
    ts_stats_info.to_pickle(r'D:/sync/Factor/v4/ts_stats_info.pkl')
    factor_train_ts.to_pickle(r'D:/sync/Factor/v4/factor_train_ts.pkl')
    factor_predict_ts.to_pickle(r'D:/sync/Factor/v4/factor_predict_ts.pkl')


