import pandas as pd
import numpy as np
import datetime,time
from datetime import timedelta
import seaborn as sns
import os
#from WindPy import w

from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller   #Dickey-Fuller test
import matplotlib.pyplot as plt
# %matplotlib inline

# Data Preprocessing
def margin(df, col, n):
    """
    Calculate marginal prices 计算因子的边际变化
    
    Args:
    df: (DataFrame) 
    price_col: (str) name of the column that represented prices.
    n: (int) time period. Usually choose 20 or 40 for emotion factors.
    
    Returns:
    df: (DataFrame) with a new column called 'change'.
    """
    whatever = (df[col] - df[col].shift(n))/df[col].shift(n)
    return whatever


def update_lag(df, yearly_factor=0, monthly_factor=0, weekly_factor=0, daily_factor=0):
    """
    考虑因子更新延迟和数据可获得性，所有数据获得时间向后延迟一期
    Args:
    df: (DataFrame) daily time-series based dataframe
    yearly_factor: (list) indices of yearly updated factor
    monthly_factor: (list) indices of monthly updated factor
    weekly_factor: (list) indices of weekly updated factor
    daily_factor: (list) indices of daily updated factor
    
    Returns:
    new_df: (DataFrame) lag-1 dataset
    """
    new_df = df.copy()
    if yearly_factor:
        for factor in yearly_factor:
            new_df[factor]=df[factor].shift(365, freq='d') 
    
    if monthly_factor:
        for factor in monthly_factor:
            new_df[factor]=df[factor].shift(30, freq='d') 

    if weekly_factor:
        for factor in weekly_factor:
            new_df[factor]=df[factor].shift(7, freq='d') 
            
    if daily_factor:
        for factor in daily_factor:
            new_df[factor]=df[factor].shift(1, freq='d')
            
    new_df=new_df.fillna(method='ffill')
    return new_df





class raw_data_clean(object):
    """
    清理数据集：删除有过多缺失值的因子，并将数据集从第一行无缺失值处切割

    Args:
    df: (DataFrame) a dataset to be cleaned
    
    Returns:
    new_df: (DataFrame) a cleaned dataset, processed by forward filling and getting rid of missing values.
    Can use get_newdata() function directly to obtain clean data.
    """
    def __init__(self, df):
        self.df = df
        self.col_names = df.columns.values.tolist()
        
    def fill_forward(self):
        self.df.index = pd.DatetimeIndex(self.df.index)
        self.dff = self.df.resample("B").ffill()
        return self.dff
    
    def count_missing(self, thre=400):
        """
        计算缺失值并去除缺失值超过阈值的因子，返还dataframe
        Args:
        df: (DataFrame) 
        thre: (int) when a column has missing value over the threshold, return the column name.

        Returns:
        df: (lst) New dataframe without columns that have too many missing values that > threshold.
        """
        df = self.fill_forward()
        print("number of missing values for each column:")
        print(df.shape[0] - df.count())
        lst = []
        for i in range(df.shape[1]):
            if df.shape[0] - df.iloc[:,i].count() > thre:
                lst.append(self.col_names[i])
        for i in lst:
            del(df[i])
        return df
    
    def get_newdata(self, thre=1500):
        """
        从第一个无缺失值的行起，截取数据表
        Args:
        df: (DataFrame) 

        Returns:
        new_df: New dataframe starting from the first row without missing value
        """
        # cut dataframe from the first row without missing value
        df = self.count_missing(thre)
        for i in range(df.shape[0]):
            if df.shape[1] - df.iloc[i,:].count() == 0:
                starting_point = i
                s_point = starting_point
                break
        new_df = df.iloc[s_point:, :]
        start_date = new_df.index[0]
        end_date = new_df.index[-1]
        print('The time span of this dataset is:\n', start_date, '-', end_date)
        return new_df

def get_correlated(df, thre=0.9):
    """
    检测相关性
    Args:
    df: (DataFrame)
    thre: (float) correlation coefficient threshold
    
    Returns:
    del_pairs: (list) a list of tuples, which are correlated factors inside.
    """
    corr = df.corr()
    corr_col = corr.columns.values.tolist()
    del_pairs = []
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if abs(corr.iloc[i,j]) >= thre and i < j:
                del_pairs.append((corr.index[i],corr_col[j]))
                print("Correlated factors are: ", corr.index[i],corr_col[j])
    return del_pairs

def test_stationarity(df):
    """
    平稳性检测
    Args:
    df: (DataFrame) with factors

    Return:
    print the result of DF test
    not_stable: (list) list of factors with negative stablility test result (not stable).
    """
    col_name = df.columns.values.tolist()
    not_stable = []
    for col in col_name:
        print('The factor is:', col)

        #Determing rolling statistics
        rolmean = df[col].rolling(window=365, center=False).mean()
        rolstd = df[col].rolling(window=365, center=False).std()

        #Plot rolling statistics:
        fig = plt.figure(figsize=(12, 8))
        orig = plt.plot(df[col], color='blue',label='Original')
        #mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        #std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title(col)
        plt.show()

        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        try:
            dftest = adfuller(df[col], autolag='AIC')  #autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
                dfoutput['Critical Value (%s)'%key] = value
            print(dfoutput)
            
            if dftest[1] > 0.01: # 0.99 confidence interval
                not_stable.append(col)
                print("\nThe series is not smooth, please do seasonal adjustment.")
        except np.linalg.linalg.LinAlgError:
            print("factor %s did not converge"%col)  

    return not_stable
	

def seasonal_adjust(not_stable_factor, df): 
    """
    季节调整
    Args:
    not_stable_factor: (list) list of factors with negative stablility test result. Obtained from the function test stationary.
    df: (DataFrame)
    new_col: (str) name of new column in dataframe
    
    Returns:
    df: (DataFrame) update dataframe with seasonal adjusted data
    """
    for i in not_stable_factor:
        df['adjusted '+ i] = df[i]-df[i].shift(1)
        df['adjusted '+ i] = df['adjusted '+ i].dropna().replace(0, method='ffill')
        del df[i]

    return df


def box_plot(df):
    """
    boxplot to check outliers distribution
    Args:
    df: (DataFrame)
    """ 

    col_list = list(df)
    for col in col_list:
        sns.boxplot(data=df[col], palette="Set1")
        print(col)
        plt.title(col)
        plt.show()


def norm_inverse(timeseries): 
    """
    Perform normal distribution inversed transformation for factors to remove outliers.
    Args:
    timeseries: (Series) a factor to be normal inverse processed, getting rid of extreme values.
    
    Returns:
    normdis_inv: (numpy.ndarray) Adjusted factor's values.
    """
    st_EF = (timeseries - np.min(timeseries)) / (np.max(timeseries)-np.min(timeseries)) # standardization
    sort_EF = st_EF.sort_values()/len(st_EF)
    bound = [] # 由于0和1不能进行逆变换处理，单独拿出来
    bound.extend([sort_EF[0], sort_EF[-1]])

    normdist_inv = np.zeros(len(st_EF))
    for i in range(len(st_EF)):
        if st_EF[i] not in bound:
            normdist_inv[i] = norm.ppf(st_EF[i])
        else:
        	normdist_inv[i] = st_EF[i]
            
    return normdist_inv


# Train-Test Split ro gernerate scrolling windows 生成滚动窗口
def get_rolling_sample(df, obs_length=730, pred_length=10):
    """
    数据集分割，以实现滚动样本预测。
    默认观察窗口长度为24个月，投资窗口长度为10个交易日。
    Args:
    df: (DataFrame) Y should be the last column. 
    obs_length: (int) length of obeservation period
    pred_length: (int) length of prediction period
    
    Returns:
    new_df: rolling samples with specific time invertals as obersavation and prediction period.
    """
    df_x = df.iloc[:,:-1]; df_y = df.iloc[:,-1]
    x = []; y = []
    time_length = df.shape[0]
    new_df = pd.DataFrame()
    
    for t in range(0, time_length, pred_length)[int(obs_length/pred_length):-1]:
        x.append(df_x[t-obs_length:t].values)
        y.append(np.sign(df_y[t+pred_length-1]-df_y[t-1]))
        
    new_df = new_df.append([x,y]).T
    new_df.columns = ["X", "Y"]

    return new_df




# factor selection
def factor_score(df, factor, obs_length=730, pred_length=10, direction=1):
    """
    根据三分位点测试因子信号。输入因子观测期、预测期和影响方向，返回因子评分。
    -1，0，1分别对应看多，看平和看空三个方向。
    ---------
    Args:
        df: (DataFrame) the last column should be asset's return rate.
        factor: (str) a column in df, meaning a specific factor
        obs_length: (int) observation period length/days
        pred_length: (int) prediction period length/days
        direction: (int) 1 or 0. Default=1, meaning good's price will be positively influenced by the value of factor.
        
    Returns:
        new_df: (DataFrame) with factor score and underlying asset's return rate.
    """
    df_size = len(df)
    df_return_rate = df.iloc[:,-2] # 涨跌幅（当日收益率）在倒数第二列
    df_swing = df.iloc[:, -1] # 振幅在倒数第一列
    
    factor_score = []; underlying_asset_return = []
    
    # Calculate [0.33, 0.67] quantile of factor
    factor_quantile = [df[factor][i-obs_length: 
                                  i].quantile([0.33, 0.67]) for i in list(range(0, df_size, pred_length)[int(obs_length/pred_length):-1])]
    factor_quantile = pd.DataFrame(factor_quantile,index=list(range(0, df_size, pred_length)[int(obs_length/pred_length):-1]))
    
    # Score for factor
    
    for i in range(0, df_size, pred_length)[int(obs_length/pred_length):-1]:
        if direction:
            if df[factor][i] >= factor_quantile[0.67][i]:
                score = 1
            elif df[factor][i] >= factor_quantile[0.33][i]:
                score = 0
            else:
                score = -1
            

        else:
            if df[factor][i] >= factor_quantile[0.67][i]:
                score = -1
            elif df[factor][i] >= factor_quantile[0.33][i]:
                score = 0
            else:
                score = 1

        factor_score.append(score)
        return_rate = sum(df_return_rate[i-1:i+pred_length-1])/pred_length
        
        
        underlying_asset_return.append(return_rate)
        
    # Concat into a dataframe
    factor_score = pd.DataFrame(factor_score, columns=['factor_score'], index=list(range(0, df_size, pred_length)[int(obs_length/pred_length):-1]))
    underlying_asset_return = pd.DataFrame(underlying_asset_return, columns=['underlying_asset_rate'], index=list(range(0, df_size, pred_length)[int(obs_length/pred_length):-1]))
    new_df = pd.concat([factor_score, underlying_asset_return], axis=1)
    time_index = df.index[list(new_df.index)]
    new_df.index = time_index
        
    return new_df


def t_signal(df):
    """
    生成单因子的三分位t统计量作为信号生成机制。
    t>1.65说明该因子预测能力显著，将该因子纳入因子库。
    --------
    Args:
    df: (DataFrame) generated by the function called factor_score
    
    Returns:
    t: (np.float64) absolute t signal value. 
    The function tests significance of factors and print the result.
    """
    long_index = df[df.factor_score == 1].index.tolist()
    short_index = df[df.factor_score == -1].index.tolist()
    
    F1 = df['underlying_asset_rate'][long_index].mean() # F1: (float)发出看多信号时未来一段/个交易日交易标的收益率的均值
    F3 = df['underlying_asset_rate'][short_index].mean() # F3: (float)发出看空信号时未来一段/个交易日交易标的收益率的均值
    S1 = df['underlying_asset_rate'][long_index].var() # S1: (float)发出看多信号时未来一段/个交易日交易标的收益率的方差
    S3 = df['underlying_asset_rate'][short_index].var() # S3: (float)发出看空信号时未来一段/个交易日交易标的收益率的方差
    n1 = len(long_index) # n1: (int)发出看多信号的样本容量
    n3 = len(short_index) # n3: (int)发出看空信号的样本容量
    print(F1,F3,S1,S3,n1, n3)
    try:
        denominator = np.sqrt((((n1-1)*S1 + (n3-1)*S3)/(n1+n3-2))*(1/n1 + 1/n3))
        t = (F1-F3)/denominator
        print("The t value of this factor is ", t)
        t = abs(t)
        return t
    
        if t > 1.65:
            print("This factor is significant.")
        else:
            print("This factor is not significant.")
        
    
    except ZeroDivisionError:
        print("This factor is not significant.")

def significant_factor(df, obs_length=360, pred_length=10):
    """
    返回显著因子
    Args:
    df: (DataFrame) with factors inside
    obs_length: (int) 
    pred_length: (int)
    
    Returns:
    significant_factor: (list) with significant factors inside
    """
    factor_list=list(df.iloc[:, :-3]) # df中后3列需为收盘价，涨跌幅和振幅
    sig_factor=[]
    for fac in factor_list:
        print("This factor is ", fac)
        df_score = factor_score(df, fac, obs_length, pred_length)
        sc = t_signal(df_score)
        if np.float64(sc) > np.float64(1.65):
            sig_factor.append(fac)

    print("\nSignificant factors are:", sig_factor)


# 权重分配
def equally_weighted(df):
    """
    等权分配
    Args:
        df: (DataFrame) with all factors' scores
        
    Returns:
        signal: (DataFrame) trading signal. positive -> long, negative -> short
    """
    weight = np.array([1/len(df.columns)]* len(df.columns))
    weighted_df = df * weight
    df["weighted_score"] = weighted_df.apply(lambda x: x.sum(),axis=1)
    
    return df


# 策略回测
def cal_profit(future_data,Signal):
    """
    计算并返回策略的累计收益；默认交易100手，佣金为0.024%, 保证金控制在15%（铁矿石保证金为10%）。
    --------
    Args:
    future_data: (DataFrame) Include datetime as index and asset price;
    Signal: (DataFrame or Array) signal generated by factors.
    
    Returns:
    profit: (list) culmulative profit at each trading time.
    Print profit and cost.
    """
    unit=100; commission=0.00024
    profit=[]; cost = 0
    data=np.array(future_data)
    #Signal=np.array(Signal)
    position={'holding':[0],'index':[]}
    index=future_data.index
    idx=[]
    
    #for i in list(range(lentest,len_backtest)):
    for i in range(len(Signal)):    
        # 开仓情况
        if position['holding'][-1] == 0 and Signal[i]>0:
            position['holding'].append(1)
            position['index'].append(i)
            pro=(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        if position['holding'][-1] == 0 and Signal[i]<0:
            position['holding'].append(-1)
            position['index'].append(i)
            pro=-(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        # 持仓情况
        if position['holding'][-1] > 0 and Signal[i]>0:
            position['holding'].append(1)
            position['index'].append(i)
            pro=(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        if position['holding'][-1] < 0 and Signal[i]<0:
            position['holding'].append(-1)
            position['index'].append(i)
            pro =-(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        # 平仓情况
        if position['holding'][-1] > 0 and Signal[i]==0:
            position['holding'].append(0)
            position['index'].append(i)
            pro=(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        if position['holding'][-1] < 0 and Signal[i]==0:
            position['holding'].append(0)
            position['index'].append(i)
            pro=-(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        # 反手情况
        if position['holding'][-1] > 0 and Signal[i] <0:
            pro =(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            position['holding'].append(-1)
            position['index'].append(i)
            cost += data[i]*unit*commission
            idx.append(index[i])
            continue

        if position['holding'][-1] < 0 and Signal[i] > 0:
            pro =-(data[i]-data[position['index'][-1]])*unit
            profit.append(pro)
            cost += data[i]*unit*commission
            position['holding'].append(1)
            position['index'].append(i)
            idx.append(index[i])
            continue

        if position['holding'][-1] == 0 and Signal[i]==0:
            profit.append(0)
        
    #pd.DataFrame(profit).cumsum().plot(color='red')
    plt.style.use('ggplot')
    plt.figure(figsize=(20,8))
    plt.plot(future_data.index,pd.DataFrame(profit).cumsum())
    plt.gcf().autofmt_xdate() # 旋转日期显示
    plt.title('Strategy Cumulative Profit')
    #plt.show()
    
    # 计算所需本金
    cul_profit = np.array(profit).cumsum()
    init_cash = future_data.max()*unit*0.15
    cul_return = cul_profit[-1]/init_cash
    
    print("total cost:", round(cost,2))
    print("profit:", profit)
    print("culmulative profit:", cul_profit)
    print("Needed initial cash is:", init_cash, "culmulative return is:", cul_return)
    return profit


# 计算最大回撤
def max_drawdown(timeseries):
    # 回撤结束时间点
    e = np.argmax(np.maximum.accumulate(timeseries) - timeseries)
    # 回撤开始的时间点
    s = np.argmax(timeseries[:e])
    return float(timeseries[s]-timeseries[e]) / timeseries[s]