import pandas as pd
from datetime import datetime
from datetime import timedelta

delta = timedelta(2)
start = datetime(2017,1,3)
print(start+2*delta)
# for i in range(1, 10):
#     now = start + i*delta
#     print(now)