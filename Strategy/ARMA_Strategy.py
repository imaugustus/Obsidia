import numpy as np
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
        self.MktData.index.to_datetime()
        self.MktData.columns.name = ['Feature', 'Code']
        self.code_first_MktData = self.MktData.swaplevel(0, 1, axis=1)

    def get_industry_descendant(self, industry_code, industry_order=1):
        stock_SWICS = self.intern['InstrumentInfo']['SWICS']
        pattern = re.compile(industry_code[0:industry_order])
        descendant = []
        for index, stock in zip(self.intern['InstrumentInfo'].index, stock_SWICS):
            if re.match(pattern, stock):
                descendant.append(index)
        return np.asarray(descendant)

    def category_index(self, industry_code, train_size=50):
        descendant = self.get_industry_descendant(industry_code, industry_order=3)
        industry_all = pd.DataFrame()
        yesterday_ret_cumsum = []
        for stock in descendant:
            try:
                industry_all[stock] = self.code_first_MktData[str(stock)]['ret'][-train_size:-1].as_matrix()
            except KeyError:
                continue
        no_NaN = industry_all.dropna(axis=1, how='any')
        industry_cumsum = no_NaN.cumsum()
        industry_index = industry_cumsum.mean(1)
        industry_index.index = self.MktData.index[-train_size:-1]
        extra_stock_performance = industry_cumsum.sub(industry_cumsum.mean(axis=1), axis=0)
        extra_stock_performance.index = self.MktData.index[-train_size:-1]
        return industry_index, extra_stock_performance, yesterday_ret_cumsum


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
        # #plt.show()
        return next_ret, arma.summary2()

    def arima_forecast(self, ts,  p, i, q,):
        arima = ARIMA(ts, order=(p, i, q)).fit(disp=-1)
        ts_predict = arima.predict()
        next_ret = arima.forecast(1)[0]
        #print("Forecast stock extra return of next day: ", next_ret)
        # plt.clf()
        # plt.plot(ts_predict, label="Predicted")
        # plt.plot(ts, label="Original")
        # plt.legend(loc="best")
        # plt.title("AR Test {},{}".format(p, q))
        # #plt.show()
        return next_ret, arima.summary2()


def main():
    test_strategy = Strategy('280201')
    industry_index, extra_stock_performance, yesterday_ret_cumsum = test_strategy.category_index(test_strategy.industry_code)
    # plt.plot(industry_index, label='Industry Index')
    # for stock in extra_stock_performance.columns:
    #     plt.plot(extra_stock_performance[stock], label=stock)
    # plt.legend(loc="best")
    # plt.show()
    # print(industry_index.shape)
    # print(extra_stock_performance.shape)
    factor = pd.Series(index=extra_stock_performance.columns)
    summary = {}
    for i, stock in enumerate(extra_stock_performance.columns):
        # predicted_extra_ret = pd.DataFrame()
        stock_predict_arma, stock_summary_arma = test_strategy.arma_forecast(extra_stock_performance[stock], 1, 0)
        summary[stock] = stock_summary_arma
        factor[i] = float(stock_predict_arma)
        # stock_predict_arima, stock_summary_arima = test_strategy.arima_forecast(extra_stock_performance[stock], 1, 0, 0)
        # if stock_predict_arma-yesterday_ret_cumsum[i] > 0:
        #     print("Predicted extra ret of ARMA is:{} bigger than 0,suggested sell out".format(stock_predict_arma-yesterday_ret_cumsum[i]))
        # else:
        #     print("Predicted extra ret of ARMA is:{} less than 0,suggested buy in".format(stock_predict_arma-yesterday_ret_cumsum[i]))
        # if stock_predict_arima-yesterday_ret_cumsum[i] > 0:
        #     print("Predicted extra ret of ARIMA is:{} bigger than 0,suggested sell out".format(stock_predict_arima-yesterday_ret_cumsum[i]), stock_predict_arima)
        # else:
        #     print("Predicted extra ret of ARIMA is:{} less than 0,suggested buy in".format(stock_predict_arima-yesterday_ret_cumsum[i]), stock_predict_arima)
    return factor, summary


if __name__ == '__main__':
    factor, summary = main()
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


