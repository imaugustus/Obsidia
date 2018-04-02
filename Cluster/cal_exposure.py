import pickle
import numpy as np


# a = np.array([[0.17, -0.05, 0.19, -0.28, 1.16], [-0.16, 0.74, 1.47, -0.59, 0.84], [-0.53, -0.24, 0.83, -0.72, 0.70]])
# print(a)
intern = pickle.load(open(r'D:/Obsidia/Data/intern.pkl', 'rb'))
# df_Cap = intern['MktCap']
# df_Data = intern['MktData']
df_Instrument = intern['InstrumentInfo']

cluster = {}
with open(r'D:/Obsidia/cluster.txt', 'r') as f:
    fo = f.readlines()
    for i in range(1, len(fo)):
        temp = fo[i].split(':')
        name = temp[0]
        code = temp[1].strip('\n').strip(' ').split(", ")
        cluster[name] = code

category = []
for code in cluster['Cluster 57']:
    index = df_Instrument.index.get_loc(code)
    category.append(df_Instrument.iloc[index, 0])

print(len(cluster['Cluster 57']), len(category))