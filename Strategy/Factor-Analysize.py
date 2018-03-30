import numpy as np
import pandas as pd
import Strategy.ARMA_Strategy
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import re


class RegressionTest:
    def __init__(self):
        self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        self.InstrumentInfo = self.intern['InstrumentInfo']
        self.MktData = self.intern['MktData']
        self.MktData.columns.name = ['Feature', 'Code']
        #self.MktData.index.to_datetime()
        self.Code_First_MktData = self.MktData.swaplevel(0, 1, axis=1)

    def drop_st(self):
        dropped_code = []
        pattern = re.compile(r"\*ST")
        for code, name in zip(self.InstrumentInfo.index, self.InstrumentInfo['Name']):
            if re.match(pattern, name):
                dropped_code.append(code)
        no_st_MktData = self.Code_First_MktData.drop(dropped_code, axis="columns", level=0)
        return no_st_MktData

    def cross_section_selection(self):
        month_end_loc = self.drop_st().index
        print(month_end_loc.is_month_end())

def main():
    test = RegressionTest()
    test.cross_section_selection()
    #print(dropped_code)

if __name__ == '__main__':
    main()