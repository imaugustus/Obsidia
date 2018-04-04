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



class Strategy:
    def __init__(self, industry_code='720000', start='2017-01-03', time_period=20, industry_order=3):
        self.industry_code = industry_code
        self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        self.MktData = self.intern['MktData']
        self.InstrumentInfo = self.intern['InstrumentInfo']
        self.First_Tradiung_Date = self.intern['InstrumentInfo']['FirstTradingDate']
        self.code_first_MktData = self.MktData.swaplevel(0, 1, axis=1)
        self.no_st_code_first_MktData, self.dropped_code = self.drop_st()
        self.start = start
        self.time_period = time_period
        self.industry_order = industry_order

    def get_industry_descendant(self, industry_code, industry_order=3):
        no_st_InstrumentInfo = self.InstrumentInfo.drop(self.dropped_code, axis="index")
        pattern = re.compile(industry_code[0:industry_order])
        descendant = []
        for index, stock in zip(no_st_InstrumentInfo.index, no_st_InstrumentInfo['SWICS']):
            if re.match(pattern, stock):
                descendant.append(index)
        return np.asarray(descendant)

    def cal_valid_start_trading_day(self, industry_stock):
        valid_start_day = {}
        for stock in industry_stock:
            start_trading_day = self.First_Tradiung_Date[stock]
            stock_timestamp = self.no_st_code_first_MktData.loc[start_trading_day, stock].name
            index = self.no_st_code_first_MktData.index.get_loc(stock_timestamp)
            valid_start_trading_index = index+60 #以开盘日60天之后作为数据起始点
            valid_start_trading_day = self.no_st_code_first_MktData.ix[valid_start_trading_index, stock].name
            valid_start_day[stock] = valid_start_trading_day
        return valid_start_day

    def drop_st(self):
        dropped_code = []
        pattern = re.compile(r"\*ST")
        for code, name in zip(self.InstrumentInfo.index, self.InstrumentInfo['Name']):
            if re.match(pattern, name):
                dropped_code.append(code)
        no_st_MktData = self.code_first_MktData.drop(dropped_code, axis="columns", level=0)
        return no_st_MktData, dropped_code

    def category_index(self, industry_code, start='2017-01-03', time_period=30, industry_order=3):
        start_index = self.no_st_code_first_MktData.index.get_loc(self.no_st_code_first_MktData.loc[start].name)
        # end_index = self.no_st_code_first_MktData.index.get_loc(self.no_st_code_first_MktData.loc[end].name)
        end_index = start_index+time_period
        descendant = self.get_industry_descendant(industry_code, industry_order)
        #valid_start_day = self.cal_valid_start_trading_day(descendant)
        industry_all = pd.DataFrame()
        for stock in descendant:
            try:
                industry_all[stock] = self.no_st_code_first_MktData[stock]['ret'].iloc[start_index:end_index+1]
            except KeyError:
                continue
        na_free_industry_all = industry_all.dropna(axis=1, how='any')
        na_free_industry_all.index = self.no_st_code_first_MktData.index[start_index:end_index+1]
        na_free_industry_index = na_free_industry_all.mean(axis=1, skipna=True)
        na_free_extra_stock_performance = na_free_industry_all.sub(na_free_industry_index, axis=0)
        return na_free_industry_index, na_free_extra_stock_performance

    def cal_real_extra_performance(self, industry_code='720000', start='2017-01-03', time_period=30, industry_order=3):
        start_index = self.no_st_code_first_MktData.index.get_loc(self.no_st_code_first_MktData.loc[start].name)
        end_index = start_index + time_period
        predict_datetimne = self.no_st_code_first_MktData.index[end_index]
        real_index = start_index + time_period + 1
        descendant = self.get_industry_descendant(industry_code, industry_order)
        industry_all = pd.DataFrame()
        for stock in descendant:
            try:
                industry_all[stock] = self.no_st_code_first_MktData[stock]['ret'].iloc[start_index:end_index+1]
            except KeyError:
                continue
        na_free_industry_all = industry_all.dropna(axis=1, how='any')
        na_dropped_stock = industry_all.columns[~industry_all.columns.isin(na_free_industry_all.columns)]
        no_na_descendant = [item for item in descendant if item not in na_dropped_stock]
        industry_index = 0
        count = 0
        stock_ret = pd.Series(index=no_na_descendant)
        NaN_stock = []
        for stock in no_na_descendant:
            ret = self.no_st_code_first_MktData[stock]['ret'].iloc[real_index]
            stock_ret[stock] = ret
            if ret != ret:
                NaN_stock.append(stock)
            else:
                count += 1
                industry_index += ret
        real_extra_performace = stock_ret - industry_index/count
        return NaN_stock, real_extra_performace, predict_datetimne

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

    def train_forecast(self):
        industry_index, extra_stock_performance = self.category_index(industry_code=self.industry_code,\
                                        start=self.start, time_period=self.time_period, industry_order=self.industry_order)
        NaN_stock, real_extra_performance, predict_datetime = self.cal_real_extra_performance(self.industry_code, start=self.start,\
                                                    time_period=self.time_period, industry_order=self.industry_order)
        predict_factor = pd.DataFrame(index=extra_stock_performance.columns, columns=['Estimated', 'Real', 'Rightness'],dtype='float')
        strategy_summary = {}
        for i, stock in enumerate(extra_stock_performance.columns):
            predict_extra_performance, stock_summary = self.arma_forecast(extra_stock_performance[stock], 1, 0)
            strategy_summary[stock] = stock_summary
            predict_factor.loc[stock, 'Estimated'] = float(predict_extra_performance)
            predict_factor.loc[stock, 'Real'] = real_extra_performance[stock]
            predict_factor.loc[stock, 'Rightness'] = 1 if float(predict_extra_performance)*real_extra_performance[stock] > 0 else -1
        return predict_factor, strategy_summary, predict_datetime, NaN_stock


if __name__ == '__main__':
    # ts_stats_info = {}
    ts_stats_info = pd.DataFrame(columns=['Square_loss_sum', 'Corr', 'Precision', 'Mean', 'Std', 'Skew', 'Kurt'], dtype='float')
    industry_code_i = '720000'
    time_period_i = 30
    industry_order_i = 3
    rng = pd.date_range(start='2017-10-01', end='2018-01-01', freq='1D')
    delta = timedelta(30)
    predict_start = datetime(2017, 10, 1) + delta
    predict_end = datetime(2018, 1, 1) + delta
    rng2 = pd.date_range(start=predict_start, end=predict_end, freq='1D')
    # ts = pd.Series(rng)
    # ts2 = pd.Series(rng2)
    test = Strategy(industry_code_i, '2017-10-01', time_period_i, industry_order_i)
    test_descendant = test.get_industry_descendant(industry_code=industry_code_i, industry_order=3)
    # iterables = [test_descendant, ['Estimated', 'Real']]
    # col = pd.MultiIndex.from_product(iterables, names=['Stock', 'Ret'])
    factor_ts = pd.DataFrame(columns=test_descendant, index=rng2)
    factor_real_ts = pd.DataFrame(columns=test_descendant, index=rng2)
    for j in range(len(rng)):
        start_date = rng[j]
        predict_date = rng2[j]
        try:
            strategy = Strategy(industry_code_i, start_date, time_period_i, industry_order_i)
            factor, summary, predict_datetime, NaN_stock = strategy.train_forecast()
            for stock in factor.index:
                try:
                    factor_ts.loc[predict_date] = factor['Estimated']
                    factor_real_ts.loc[predict_date] = factor['Real']
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
            ts_stats_info.loc[predict_datetime, 'Square_loss_sum'] = squre_loss_sum
            ts_stats_info.loc[predict_datetime, 'Corr'] = corr.iloc[0, 1]
            ts_stats_info.loc[predict_datetime, 'Precision'] = Precision
            ts_stats_info.loc[predict_datetime, 'Mean'] = Mean
            ts_stats_info.loc[predict_datetime, 'Std'] = Std
            ts_stats_info.loc[predict_datetime, 'Skew'] = Skew
            ts_stats_info.loc[predict_datetime, 'Kurt'] = Kurt
        except KeyError:
            continue
    factor_ts = factor_ts.dropna(axis=0, how='all')
    factor_real_ts = factor_real_ts.dropna(axis=0, how='all')
    ts_stats_info.to_pickle(r'D:/sync/Factor/ts_stats_info.pkl')
    factor_ts.to_pickle(r'D:/sync/Factor/factor_ts.pkl')
    factor_real_ts.to_pickle(r'D:/sync/Factor/factor_real_ts.pkl')
    fig = plt.figure()
    for index, item in enumerate(ts_stats_info.columns):
        fig.add_subplot(1, 6, index+1)
        ts_stats_info[item].plot()
        plt.show()
        # factor.to_csv(r'D:/sync/Factor/factor.csv')

    # x = factor.index
    # y = factor
    # #fig = plt.figure(figsize=(15,2))
    # plt.scatter(x, y)
    # plt.plot([x[0],x[-1]], [0,0], color='r')
    # for a, b in zip(x, y):
    #     plt.text(a, b + 0.01, '{0:.5}'.format(b), ha='center', va='bottom', fontsize=7)
    # plt.title("280000")
    # plt.xlabel("Stock Code")
    # plt.ylabel("Diviation of each stock from index")
    # plt.xticks([])
    # plt.show()

    # temp = np.array(ret)
    # t = statsmodels.tsa.stattools.adfuller(temp)  # ADF检验
    # output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
    # output['value']['Test Statistic Value'] = t[0]
    # output['value']['p-value'] = t[1]
    # output['value']['Lags Used'] = t[2]
    # output['value']['Number of Observations Used'] = t[3]
    # output['value']['Critical Value(1%)'] = t[4]['1%']
    # output['value']['Critical Value(5%)'] = t[4]['5%']
    # output['value']['Critical Value(10%)'] = t[4]['10%']
    # print(output)

    import statsmodels.api as sm
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='bic')['bic_min_order']  # BIC
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='hqic')['hqic_min_order'] # HQIC

    # order = (1, 1)
    # train = ret[:-50]
    # test = ret[-50:]
    # tempModel = sm.tsa.ARMA(train, order).fit()
    # tempModel.summary2()


