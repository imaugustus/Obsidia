import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


def pkl_to_csv(pkl_path, csv_path):
    df_set = pd.read_pickle(pkl_path)
    for key in df_set.keys():
        output_path = os.path.join(csv_path, str(key) + r'.csv')
        df_set[key].to_csv(output_path)

def description(pkl_path):
    df_set = pd.read_pickle(pkl_path)
    for key in df_set.keys():
        #print(type(df_set[key]))
        # return df_set[key].shape
        print(df_set[key].shape)

def plot_cap_history(cap_csv_path):
    """
    共有1705支股票，803天交易数据
    :param cap_csv_path:
    :return:
    """
    df = pd.read_csv(cap_csv_path)
    print("shape-----", df.shape)
    df_example = df.iloc[:, 1:5]
    df_example.plot()
    plt.show()

def plot_data_history(data_csv_path):
    df = pd.read_csv(data_csv_path)
    print("Shape------", df.shape)
    # print("Head-------", df.head(5))
    df.iloc[:, 0:7].plot()
    df.iloc[:, 7:10].plot()
    # turnover = np.array(df.iloc[:, 7])
    # volume = np.array(df.iloc[:, 8])
    # vwap_real = np.array(df.iloc[:, 9])
    # vwap_est = turnover/volume
    # plt.plot(pd.DataFrame({"vwap_real": vwap_real, "vwap_est": vwap_est}))
    plt.show()

def get_all_pkl(pkl_path=r'D:/Obsidia/Data/intern.pkl'):
    intern = pickle.load(open(pkl_path, 'rb'))
    return intern
