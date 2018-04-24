
1. 横截面回归：lag0: t日的相对收益与t-1日收盘计算的因子暴露做回归，滚动计算得到每一期回归的因子收益率、r-square。lag1：t日的相对收益与t-2日收盘计算的因子暴露做回归，得到结果同上，横轴为截面日期，纵轴为累计因子收益作图(cumsum)，并保存每期因子收益，r-square。Lag2、3、4同上。（可以把lag0-4，5条曲线作在一张图中）  
2. 截面相关系数：lag0: t日的相对收益与t-1日收盘计算的因子暴露的相关系数，滚动计算得到每一期的相关系数用spearman法。lag1-4同上。作图横轴为日期，纵轴为因子相关性的cumsum。（可以把lag0-4，5条曲线作在一张图中）  
3. 分层测试：lag0: t日按照t-1日收盘因子暴露大小分为5（或10）层，得到t日每层股票的绝对/相对收益的平均值，滚动计算，得到每期各层的绝对收益/相对收益，分别累加计算。作图：绝对收益图，横轴为时间，纵轴为每层的绝对收益累加。相对收益图，横轴为时间，纵轴为每层的相对收益累加。再做一个lag1的结果。