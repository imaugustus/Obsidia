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
from statsmodels.tsa.arima_model import  ARMA
import re
import util
import numpy


class Strategy:
    def __init__(self, industry_code):
        self.industry_code = industry_code
        self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        self.MktData = self.intern['MktData']
        self.MktData.index.to_datetime()
        self.MktData.columns.name = ['Feature', 'Code']
        self.code_first_MktData = self.MktData.swaplevel(0, 1, axis=1)

    def get_industry_descendant(self, industry_code):
        stock_SWICS = self.intern['InstrumentInfo']['SWICS']
        pattern = re.compile(industry_code[0:2])
        descendant = []
        for index, stock in zip(self.intern['InstrumentInfo'].index, stock_SWICS):
            if re.match(pattern, stock):
                descendant.append(index)
        return np.asarray(descendant)

    def check_stock(self, descendant):
        return

    def category_index(self, industry_code):
        descendant = self.get_industry_descendant(industry_code)
        industry_all = pd.DataFrame()
        real_descendant = []
        for stock in descendant:
            try:
                industry_all[stock] = self.code_first_MktData[str(stock)]['close'].as_matrix()
                real_descendant.append(stock)
            except KeyError:
                continue
        industry_index = industry_all.mean(axis=1)
        industry_index.index = self.MktData.index
        return industry_index

    def arma_forecast(self, ts,  p, q,):
        arma = ARMA(ts, order=(p, q)).fit(disp=-1)
        ts_predict = arma.predict()
        next_price = arma.forecast(1)[0]
        print("Forecast stock price of next day: ", next_price)
        plt.clf()
        plt.plot(ts_predict, label="Predicted")
        plt.plot(ts, label="Original")
        plt.legend(loc="best")
        plt.title("AR Test {},{}".format(p, q))
        plt.show()
        return next_price, arma.summary2()


def main():
    test_strategy = Strategy('720000')
    industry_index = test_strategy.category_index(test_strategy.industry_code)
    industry_next_price, industry_model_info = test_strategy.arma_forecast(industry_index, 1, 1)
    descendant = test_strategy.get_industry_descendant(industry_code=test_strategy.industry_code)
    for stock_code in descendant:
        try:
            stock_next_price, _ = test_strategy.arma_forecast(test_strategy.code_first_MktData[str(stock_code)]['close'].dropna(), 1, 1)
            if stock_next_price > industry_next_price:
                print("Price of {} will drop, selling is advised".format(stock_code))
            elif stock_next_price < industry_next_price:
                print("Price of {} will rise, buying is advised".format(stock_code))
            else:
                print("Hold on this stock")
        except KeyError:
            pass
        except numpy.linalg.linalg.LinAlgError:
            pass


if __name__ == '__main__':
    main()


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


