import numpy as np
import pandas as pd
import Strategy.ARMA_Strategy
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import re
import math


class Group:
    def __init__(self, industry_code):
        self.industry_code = industry_code
        self.intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
        self.MktData = self.intern['MktData']
        self.MktData = self.MktData.swaplevel(0, 1, axis=1)
        self.factor = pickle.load(open(r'D:/sync/Factor/factor_real_ts.pkl', 'rb'))
        self.factor = self.factor.dropna(axis=1, how='any')

    def get_group_section(self):
        temp = []
        for i in range(0, len(self.factor.index), 10):
            temp.append(self.factor.iloc[i, :])
        return pd.concat(temp, axis=1).transpose()

    def factor_classification(self, section):
        sorted_section = section.sort_values(ascending=False)
        group_length = math.ceil(len(sorted_section)/5)
        classification_index = {}
        for i in range(5):
            if i == 4:
                classification_index[i+1] = sorted_section.index[i*group_length, len(sorted_section)]
            else:
                classification_index[i+1] = sorted_section.index[i*group_length, (i+1)*group_length]
        return classification_index

    def classification_all_section(self):
        sections = self.get_group_section()
        result = {}
        for section in sections.index:
            result[section] = self.factor_classification(sections.loc[section, :])



def main():
    test = Group('720000')
    return test.factor


if __name__ == '__main__':
    result = main()