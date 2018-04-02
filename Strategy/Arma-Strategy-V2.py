import numpy as np
import time
import datetime
import pandas as pd
from datetime import datetime
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


class Strategy:
    def __init__(self, industry_code):
        self.industry_code = industry_code
        self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        self.MktData = self.intern['MktData']
        self.InstrumentInfo = self.intern['InstrumentInfo']
        self.First_Tradiung_Date = self.intern['InstrumentInfo']['FirstTradingDate']
        self.code_first_MktData = self.MktData.swaplevel(0, 1, axis=1)
        self.no_st_code_first_MktData, self.dropped_code = self.drop_st()

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

    def category_index(self, industry_code, start='2018-02-01', end='2018-03-14', industry_order=3):
        start_index = self.no_st_code_first_MktData.index.get_loc(self.no_st_code_first_MktData.loc[start].name)
        end_index = self.no_st_code_first_MktData.index.get_loc(self.no_st_code_first_MktData.loc[end].name)
        descendant = self.get_industry_descendant(industry_code, industry_order=3)
        print("Category:", descendant.shape)
        #valid_start_day = self.cal_valid_start_trading_day(descendant)
        industry_all = pd.DataFrame()
        for stock in descendant:
            try:
                industry_all[stock] = self.no_st_code_first_MktData[stock]['ret'][start:end].as_matrix()
            except KeyError:
                continue
        na_industry_all = industry_all.dropna(axis=1, how='any')
        na_industry_all.index = self.no_st_code_first_MktData.index[start_index:end_index+1]
        na_industry_index = na_industry_all.mean(axis=1, skipna=True)
        na_extra_stock_performance = na_industry_all.sub(na_industry_index, axis=0)
        return na_industry_index, na_extra_stock_performance

    def cal_real_extra_performance(self, industry_code, date='2018-03-15', industry_order=3):
        descendant = self.get_industry_descendant(industry_code, industry_order=3)
        print("Next:", descendant.shape)
        industry_index = 0
        count = 0
        stock_ret = pd.Series(index=descendant)
        for stock in descendant:
            count += 1
            ret = self.no_st_code_first_MktData[stock]['ret'].loc[date]
            print(ret)
            industry_index += ret
            stock_ret[stock] = ret
        real_extra_performace = stock_ret - industry_index/count
        return real_extra_performace

    def arma_forecast(self, ts,  p, q,):
        arma = ARMA(ts, order=(p, q)).fit(disp=-1)
        ts_predict = arma.predict()
        next_ret = arma.forecast(1)[0]
        #print("Forecast stock extra return of next day: ", next_ret)
        # plt.clf()
        # plt.plot(ts_predict, label="Predicted")
        # plt.plot(ts, label="Original")
        # plt.legend(loc="best")
        # plt.title("AR Test {},{}".format(p, q))
        # plt.show()
        return next_ret, arma.summary2()

    def arima_forecast(self, ts,  p, i, q,):
        arima = ARIMA(ts, order=(p, i, q)).fit(disp=-1)
        ts_predict = arima.predict()
        next_ret = arima.forecast(1)[0]
        return next_ret, arima.summary2()


def train_forecast():
    strategy = Strategy('720000')
    industry_index, extra_stock_performance = strategy.category_index(industry_code=\
        strategy.industry_code, start='2018-02-01', end='2018-03-14', industry_order=2)
    strategy_factor = pd.DataFrame(index=extra_stock_performance.columns, columns=['Estimated', 'Real'])
    real_extra_performance = strategy.cal_real_extra_performance(industry_code=\
        strategy.industry_code, date='2018-03-15', industry_order=3)
    print(real_extra_performance)
    strategy_summary = {}
    for i, stock in enumerate(extra_stock_performance.columns):
        predict_extra_performance, stock_summary = strategy.arma_forecast(extra_stock_performance[stock], 1, 0)
        strategy_summary[stock] = stock_summary
        strategy_factor.loc[stock, 'Estimated'] = float(predict_extra_performance)
        strategy_factor.loc[stock, 'Real'] = real_extra_performance[stock]
    return strategy_factor, strategy_summary




if __name__ == '__main__':
    factor, summary = train_forecast()
    # factor, summary = main()
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

    #
    import statsmodels.api as sm
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='bic')['bic_min_order']  # BIC
    # sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='hqic')['hqic_min_order'] # HQIC

    # order = (1, 1)
    # train = ret[:-50]
    # test = ret[-50:]
    # tempModel = sm.tsa.ARMA(train, order).fit()
    # tempModel.summary2()


