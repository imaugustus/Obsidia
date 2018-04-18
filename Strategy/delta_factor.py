import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
InstrumentInfo = intern['InstrumentInfo']
MktData = MktData.swaplevel(0, 1, axis='columns')


def get_ret():
    ret = MktData.loc[:, (slice(None), 'ret')]
    ret.columns = ret.columns.droplevel(1)
    return ret


def get_industry_descendant(industry_code, descendant_order):
    pattern = re.compile(industry_code[0:3+descendant_order])
    descendant = []
    for stock_code, representation in zip(InstrumentInfo.index, InstrumentInfo['SWICS']):
        if re.match(pattern, representation):
            descendant.append(stock_code)
    return descendant


def get_industry_delta_factor(ret, industry_code, descendant_order):
    descendant = get_industry_descendant(industry_code, descendant_order)
    industry_factor_average = ret.loc[:, descendant].mean(axis=0)
    # print(industry_factor_average)
    industry_delta_factor = ret.loc[:, descendant] - industry_factor_average
    return industry_delta_factor


if __name__ == '__main__':
    test_ret = get_ret()
    all_industry_code = []
    identities = []
    for swics in InstrumentInfo['SWICS']:
        identity = swics[0:3]
        if identity in identities:
            continue
        else:
            identities.append(identity)
            all_industry_code.append(identity + r'000')
    for industry_code in all_industry_code:
        industry_delta_factor = get_industry_delta_factor(test_ret, industry_code, 0)
        industry_delta_factor.to_pickle(r'D:/sync/Factor/v5/delta_factor_{}.pkl'.format(industry_code))
