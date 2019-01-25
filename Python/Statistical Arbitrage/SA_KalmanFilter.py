# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:29:03 2018

@author: rbgud
"""

from pykalman import KalmanFilter
import ffn
import matplotlib.cm as cm
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from scipy import poly1d
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sys
sns.set()

#%% Data Load

def get_data(file_name):
    
    data = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/통계적차익거래기법/{}'.format(file_name),
                                                                           na_values=['#N/A'],engine='python')
    
    data.index = pd.to_datetime(data['Date'],format = '%Y-%m-%d')
    data.pop('Date')    
       
    return data

Close = get_data('close.csv')
Open = get_data('open.csv')
Sma5 = get_data('sma5.csv')
Sma14 = get_data('sma14.csv')
Wma = get_data('wma.csv')
Rsi = get_data('rsi.csv')
Mfi5 = get_data('MFI5.csv')
Mfi14 = get_data('MFI14.csv')
volume = get_data('volume.csv')

returns = Close.pct_change()
returns = returns.iloc[1:,:]

# PCA 분석을 위한 formation period return 저장

start_date = returns.index[0]
end_date = returns.index[-1]

returns['date_count_index'] = np.arange(len(returns))

formation_period_index = returns.iloc[:int(len(returns)*0.8),:].index[:]
trade_period_index = returns.iloc[int(len(returns)*0.8):,:].index[:] 

returns = returns.ix[formation_period_index] 

returns=returns.dropna(axis=1,how='any')

#%% 과거 수익률을 이용하여 PCA를 통해 현재 시점의 리턴에 대한 설명력 측정 

'''100개의 component를 사용하여 확인'''

X=returns.values
X=scale(X)

pca = PCA(n_components=100)

pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.title('PCA explained variance ratio')
plt.ylabel('Explained Percentage(%)')
plt.xlabel('Number of components')
plt.show()

#%% 
''' PCA , DBSCAN으로 클러스터링 (후보군 찾기) '''


N_PRIN_COMPONENTS = 30
pca = PCA(n_components=N_PRIN_COMPONENTS)
pca.fit(returns)

pca.components_.T.shape

X=np.array(pca.components_.T)
X = preprocessing.StandardScaler().fit_transform(X)


clf = DBSCAN(eps=1.2, min_samples=3)


clf.fit(X)
labels = clf.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   #숫자 5, 총 5개의 클러스터 있음

print ("\nClusters discovered: %d" % n_clusters_)

clustered = clf.labels_   #{-1,-1,.....0....4... } 총 200개 사이즈

ticker_count = len(returns.columns) 


#%%
# clustered_series의 값이 같은 것들끼리 동일 cluster로 분류된 주식들을 의미함
clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series_all = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series = clustered_series[clustered_series != -1]


CLUSTER_SIZE_LIMIT = 9999

#한 클러스터에 몇 개의 주식이 있는지 확인 
counts = clustered_series.value_counts() 

# ticker_count_reduced 는 cluster안에 주식이 1개 이상 9999개 이하인 경우의 종목개수를 카운트
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]

print("Clusters formed : %s " % len(ticker_count_reduced))
print ("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())
#%% 차원 축소하여 Cluster plot을 출력

X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)

plt.figure(1, facecolor='white')
plt.clf()
plt.axis('off')

plt.scatter(
    X_tsne[(labels!=-1), 0],
    X_tsne[(labels!=-1), 1],
    s=100,
    alpha=0.85,
    c=labels[labels!=-1],
    cmap=cm.Paired)

plt.scatter(
    X_tsne[(clustered_series_all==-1).values, 0],
    X_tsne[(clustered_series_all==-1).values, 1],
    s=100,
    alpha=0.05
)

plt.title('T-SNE of all Stocks with DBSCAN Clusters Noted');


#%% Cluster Member Count 하여 bar plot 출력

plt.barh(range(len(clustered_series.value_counts())),clustered_series.value_counts())
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number');

# Cluster에 존재하는 종목들의 standardized lof price plot 출력
cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]  

for clust in cluster_vis_list[0:len(cluster_vis_list)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(Close[tickers].mean())
    data = np.log(Close[tickers]).sub(means)
    data.plot(title='Stock Time Series for Cluster %d' % clust)
    
#%% Cointegrate Test를 통한 Pair 선택
    
''' Cointegrate 이용한 Pair 찾기 (Mean - reverting 찾기)'''

def find_cointegrated_pairs(dataframe, critial_level = 0.05): 
    n = dataframe.shape[1] # the length of dateframe
    pvalue_matrix = np.ones((n, n)) # initialize the matrix of p
    keys = dataframe.keys() # get the column names
    pairs = [] # initilize the list for cointegration
    for i in range(n):
        for j in range(i+1, n): # for j bigger than i
            stock1 = dataframe[keys[i]] # obtain the price of two contract
            stock2 = dataframe[keys[j]]
            result = sm.tsa.stattools.coint(stock1, stock2) # get conintegration
            pvalue = result[1] # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level: # if p-value less than the critical level 
                pairs.append((keys[i], keys[j], pvalue)) # record the contract with that p-value
             
    return pvalue_matrix, pairs


cluster_tickers = []
for i in range(len(clustered_series)):
    
    make_clusters_list = clustered_series.index[i]
    cluster_tickers.append(make_clusters_list)
    
    

all = Close[cluster_tickers].dropna().copy()
all.head()
    
#fig = plt.figure(figsize=(15,8))
pvalues, pairs = find_cointegrated_pairs(all.dropna(),0.05)
sns.heatmap(1-pvalues, xticklabels=Close[clustered_series.index].columns, yticklabels=Close[clustered_series.index].columns, cmap='RdYlGn_r', mask = (pvalues == 1))
p = pd.DataFrame(pairs,columns=['S1','S2','Pvalue'])
p_sorted = p.sort_index(by='Pvalue').reset_index(drop=True)   
p_sorted_copy = p_sorted

#%%

def print_func(p,all):
    
    a = []
    for i in range(p.shape[0]):
        
        s1 = p.iloc[i][0]
        s2 = p.iloc[i][1]
#        print (i,s1,s2)
               
        stock_df = all[s1]
#        stock_df_mean = np.log(stock_df).mean()
#        stock_df_ = np.log(all[s1]).sub(stock_df_mean)
        
        stock_df2 = all[s2]
#        stock_df2_mean = np.log(stock_df2).mean()
#        stock_df2_ = np.log(all[s2]).sub(stock_df2_mean)
        print("corr : ",np.corrcoef(stock_df,stock_df2)[0,1] )
        a_ = np.corrcoef(stock_df,stock_df2)[0,1]
        a.append(a_)
#        fig = plt.figure(figsize=(12,8))
#        stock_df_.plot(color='#F4718B')
#        stock_df2_.plot(color='#407CE2')
#        plt.xlabel("Time"); plt.ylabel("Price")
#        plt.legend([s1, s2])
#        plt.show()
    a=pd.DataFrame(a)    
    return a
        
def print_func_scatter(p,all):
    
    for i in range(p.shape[0]):
        
        s1 = p.iloc[i][0]
        s2 = p.iloc[i][1]
        #print (i,s1,s2)
        
        stock_df = all[s1]
        stock_df2 = all[s2]    
        fig = plt.figure(figsize=(12,8))
        plt.scatter(stock_df,stock_df2)
        plt.xlabel(s1); plt.ylabel(s2)
        plt.show()        
 
#%%       
p_sorted_ = print_func(p_sorted, all)    
p_sorted2 = pd.concat([p_sorted,p_sorted_],axis=1) 
p_sorted_copy = p_sorted.loc[p_sorted2[0]>0.90].reset_index(drop=True)
#print_func_scatter(p_sorted, all) 

#%%

def KalmanFilterAverage(x):
    # Construct a Kalman filter
   
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

#  Kalman filter regression
def KalmanFilterRegression(x,y):

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                      initial_state_mean=[0,0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)
    
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means

#def half_life(spread):
#  spread_lag = spread.shift(1)
#  spread_lag.iloc[0] = spread_lag.iloc[1]
#  spread_ret = spread - spread_lag
#  spread_ret.iloc[0] = spread_ret.iloc[1]
#  spread_lag2 = sm.add_constant(spread_lag)
#  model = sm.OLS(spread_ret,spread_lag2)
#  res = model.fit()
#  halflife = int(round(-np.log(2) / res.params[1],0))
#
#  if halflife <= 0:
#    halflife = 1
#  return halflife    

def zScore_distribution_hist(s1, s2, x, y ):
    
    df = pd.DataFrame({'y':y,'x':x})
    state_means = KalmanFilterRegression(KalmanFilterAverage(x),KalmanFilterAverage(y))
    
    df['spread'] = state_means[:,0]
    
    # calculate z-score with window 
    
    meanSpread = df.spread.mean()
    stdSpread = df.spread.std()
    df['zScore'] = ((df.spread-meanSpread)/stdSpread).rolling(window=10).mean()
    df.dropna(axis=0)
    plt.figure(figsize = (12,7))
    df['zScore'].hist(bins=30)
    plt.title('Distribution of zScore')
    plt.grid(True)
    plt.show()
    print("upper 10% :{} , lower 10% :{}".format(df.zScore.quantile(0.9),df.zScore.quantile(0.1)))
    
    return

#%%
    
test_start = '2017-09-14'
    
for i in range(len(pairs)):

    s1 = pairs[0][0]
    s2 = pairs[0][1]
    x = Close.loc[:test_start][pairs[i][0]].dropna()
    y = Close.loc[:test_start][pairs[i][1]].dropna()   
    zScore_distribution_hist(s1,s2,x,y)

#%%
# loss cut

def stop_loss(df,loss_cut):
    
    if loss_cut == None:
        
        df = df
        
    else:
          
        for i in range(1,len(df)-1):
            
            if ((df.iloc[i]['port rets'] < loss_cut) * (df.iloc[i]['numUnits']==df.iloc[i-1]['numUnits']) 
                *( df.iloc[i]['numUnits'] == df.iloc[i+1]['numUnits'])) == True:
          
                df.iloc[i+1]['port rets'] = 0
                
            elif ((df.iloc[i]['port rets'] == 0) * (df.iloc[i]['numUnits']==df['numUnits'][i-1]) 
                *( df.iloc[i]['port rets'] == df.iloc[i+1]['numUnits'])) == True:
                
                df.iloc[i+1]['port rets'] = 0
             
    return df    
    
#%% Just Using spread
def backtest(s1, s2, x, y,loss_cut,upper_q,lower_q):
    #############################################################
    # INPUT:
    # s1: the symbol of contract one
    # s2: the symbol of contract two
    # x: the price series of contract one
    # y: the price series of contract two
    # OUTPUT:
    # df['cum rets']: cumulative returns in pandas data frame
    # sharpe: sharpe ratio
    # CAGR: CAGR
    
    # run regress-ion to find hedge ratio and then create spread series
    df = pd.DataFrame({'y':y,'x':x})
    state_means = KalmanFilterRegression(KalmanFilterAverage(x),KalmanFilterAverage(y))
    
    df['hr'] = state_means[:,0]
    df['intercept'] = state_means[:,1]
    df['spread'] = df.y - (df.x * df.hr)- df.intercept

    #calculate z-score with window 

    meanSpread = df.spread.mean()
    stdSpread = df.spread.std()
    df['zScore'] = ((df.spread-meanSpread)/stdSpread).rolling(window=10).mean()

    ##############################################################
    # trading logic 
    
    l_entryZscore = df.zScore.quantile(lower_q)
    s_entryZscore = df.zScore.quantile(upper_q)
    l_exitZscore = 0
    s_exitZscore = 0
   
    #set up num units long
    df['long entry'] = ((df.zScore < l_entryZscore) & ( df.zScore.shift(1) > l_entryZscore))
    df['long exit'] = ((df.zScore > l_exitZscore) & (df.zScore.shift(1) < l_exitZscore) )
    df['num units long'] = np.nan 
    df.loc[df['long entry'],'num units long'] = 1 
    df.loc[df['long exit'],'num units long'] = 0 
    df['num units long'][0] = 0 
    df['num units long'] = df['num units long'].fillna(method='pad')
    
    #set up num units short 
    df['short entry'] = ((df.zScore > s_entryZscore) & ( df.zScore.shift(1) < s_entryZscore))
    df['short exit'] = ((df.zScore < s_exitZscore) & (df.zScore.shift(1) > s_exitZscore))
    df.loc[df['short entry'],'num units short'] = -1
    df.loc[df['short exit'],'num units short'] = 0
    df['num units short'][0] = 0
    df['num units short'] = df['num units short'].fillna(method='pad')
    
    df['numUnits'] = df['num units long'] + df['num units short']
    
    df['cost_long'] = -0.003*(df['long entry']*1)
    df ['cost_short'] = -0.003*(df['short entry']*1)
       
    df['port rets'] = -df['numUnits'].shift(1)*(Close[pairs[i][0]].pct_change().loc[test_start:])+\
                      df['numUnits'].shift(1)*(Close[pairs[i][1]].pct_change().loc[test_start:])+\
                      (df['cost_long']+df['cost_short'])*2
             
    
    df_copy = df[['port rets','numUnits']]
    
    df['port rets SL'] = stop_loss(df_copy,loss_cut)['port rets']
    
    df['cum rets'] = df['port rets'].cumsum()
    df['cum rets'] = df['cum rets'] + 1
    df['cum rets'].dropna()
    
    df['cum rets SL'] = df['port rets SL'].cumsum()
    df['cum rets SL'] = df['cum rets SL'] + 1
    df['cum rets SL'].dropna()
    
    ##############################################################

#    signal = 1*df[['zScore','long entry','short entry','long exit','short exit']]
#    plt.figure(figsize=(15,7))
#    df['zScore'].plot()
#    plt.axhline(l_entryZscore, color='green', linestyle='--')
#    plt.axhline(s_entryZscore, color='red', linestyle='--')
#    buy_open = signal.loc[signal['long entry']==1]['zScore']
#    sell_open = signal.loc[signal['short entry']==1]['zScore']
#    buy_exit = signal.loc[signal['long exit']==1]['zScore']
#    sell_exit = signal.loc[signal['short exit']==1]['zScore']
#    buy_open.plot(color='g', linestyle='None',markersize= 8, marker='^')
#    sell_open.plot(color='r', linestyle='None',markersize= 8, marker='v')
#    buy_exit.plot(color='g', linestyle='None',markersize= 8, marker='v')
#    sell_exit.plot(color='r', linestyle='None',markersize= 8, marker='^')
#    plt.legend(['Rolling z-Score', 'Long entry criteria', 'Short entry criteria',
#                'long open','short open','long exit','short exit'])
#    plt.axhline(0, color='black')
#    plt.title('Long/Short Buy and Exit Signal with z-Score')
#    plt.xlabel('Date')
#    plt.ylabel('Rolling z-Score')
#    plt.show()
    ##############################################################
    
    try:
        sharpe = ((((df['port rets'].mean()- 0.0235/252) )/ df['port rets'].std()) * np.sqrt(252))
        sharpe_SL = (((df['port rets SL'].mean() )/ df['port rets SL'].std()) * np.sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0
        sharpe_SL = 0.0
    
    ##############################################################
    start_val = 1
    end_val = df['cum rets'].iat[-1]
    
    start_val_SL = 1
    end_val_SL = df['cum rets SL'].iat[-1]
    
    start_date = df.iloc[0].name
    end_date = df.iloc[-1].name
    days = (end_date - start_date).days
    
    CAGR = round(((float(end_val) / float(start_val)) ** (252.0/days)) - 1,4)
    CAGR_SL = round(((float(end_val_SL) / float(start_val_SL)) ** (252.0/days)) - 1,4)
    
    return df, sharpe, sharpe_SL, CAGR, CAGR_SL

#%%

pairs = [('LMT UN Equity','TRV UN Equity'),('NOC UN Equity','TRV UN Equity'),('LMT UN Equity','MMC UN Equity'), 
           ('PNC UN Equity','BBT UN Equity'),('HON UN Equity', 'DHR UN Equity')]  


for i in range (len(pairs)):
    plt.figure(figsize=(12,7))
    a = Close.loc[:][pairs[i][0]]/Close.loc[:][pairs[i][0]][0]
    b = Close.loc[:][pairs[i][1]]/Close.loc[:][pairs[i][1]][0]
    plt.plot(a.index,a,label=pairs[i][0])
    plt.plot(b.index,b,label=pairs[i][1])
    plt.legend(loc='upper left')
    plt.axvline(test_start, color='red', linestyle='--')
    plt.ylabel('Scaled Price')
    plt.xlabel('Date')
    plt.title('Scaled Price for pairs')
    plt.show()
    
#%%
# visualize the correlation between assest prices over time
plt.figure(figsize=(12,7))
dates = [str(p.date()) for p in Close.loc[test_start:][::int(len(Close.loc[test_start:])/10)].index]
colors = np.linspace(0.1, 1, len(Close.loc[test_start:]))
sc = plt.scatter(Close.loc[test_start:][pairs[0][0]], Close.loc[test_start:][pairs[0][1]], 
                 s=50, c=colors, cmap=plt.get_cmap('jet'), edgecolor='k', alpha=0.7)
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p.date()) for p in Close.loc[test_start:][::len(Close.loc[test_start:])//9].index]);

plt.xlabel(pairs[0][0])
plt.ylabel(pairs[0][1])

df = pd.DataFrame({'y':Close.loc[test_start:][pairs[0][1]],'x':Close.loc[test_start:][pairs[0][0]]})
state_means = KalmanFilterRegression(KalmanFilterAverage(df.x),KalmanFilterAverage(df.y))
# add regression lines
step = 25
xi = np.linspace(Close.loc[test_start:][pairs[0][0]].min(), Close.loc[test_start:][pairs[0][0]].max(), 2)
colors_l = np.linspace(0.1, 1, len(state_means[::step]))

for i, b in enumerate(state_means[::step]):
    plt.plot(xi, b[0] * xi + b[1], alpha=.5, lw=2, c=plt.get_cmap('jet')(colors_l[i]))    

plt.title('Stock prices and Kalman Filter regression line')

#%% 

loss_cut=None
num_Pairs = len(pairs)
loss_cut = -0.01
    
port_ret = pd.DataFrame()
port_ret2 = pd.DataFrame()
port_ret3 = pd.DataFrame()

port_ret_with_stoploss = pd.DataFrame()
port_ret_with_stoploss2 = pd.DataFrame()
port_ret_with_stoploss3 = pd.DataFrame()

for i in range(num_Pairs):
    port = pd.DataFrame()
    port2 = pd.DataFrame()
    port3 = pd.DataFrame()
    stoploss_port_ = pd.DataFrame()
    stoploss_port_2 = pd.DataFrame()
    stoploss_port_3 = pd.DataFrame()
    
    df, sharpe, sharpe_SL, CAGR, CAGR_SL = backtest(pairs[i][0], pairs[i][1], 
                                                    Close.loc[test_start:][pairs[i][0]],
                                                    Close.loc[test_start:][pairs[i][1]],
                                                    loss_cut,upper_q=0.8,lower_q=0.2)
    
    df2, sharpe2, sharpe_SL2, CAGR2, CAGR_SL2 = backtest(pairs[i][0], pairs[i][1], 
                                                    Close.loc[test_start:][pairs[i][0]],
                                                    Close.loc[test_start:][pairs[i][1]],
                                                    loss_cut,upper_q=0.9,lower_q=0.1)
    
    df3, sharpe3, sharpe_SL3, CAGR3, CAGR_SL3 = backtest(pairs[i][0], pairs[i][1], 
                                                    Close.loc[test_start:][pairs[i][0]],
                                                    Close.loc[test_start:][pairs[i][1]],
                                                    loss_cut,upper_q=0.7,lower_q=0.3)
        
    stoploss_port = stop_loss(df[['port rets','numUnits']],loss_cut=loss_cut)
    
    stoploss_port_ = stoploss_port['port rets']
    
    stoploss_port_.name ='Pair{}'.format(i+1)
    
    port_ret_with_stoploss = pd.concat([port_ret_with_stoploss,stoploss_port_],axis=1)
    
    stoploss_port2 = stop_loss(df2[['port rets','numUnits']],loss_cut=loss_cut)
    
    stoploss_port_2 = stoploss_port2['port rets']
    
    stoploss_port_2.name ='Pair{}'.format(i+1)
    
    port_ret_with_stoploss2 = pd.concat([port_ret_with_stoploss2,stoploss_port_2],axis=1)
    
    stoploss_port3 = stop_loss(df3[['port rets','numUnits']],loss_cut=loss_cut)
    
    stoploss_port_3 = stoploss_port3['port rets']
    
    stoploss_port_3.name ='Pair{}'.format(i+1)
    
    port_ret_with_stoploss3 = pd.concat([port_ret_with_stoploss3,stoploss_port_3],axis=1)
    
    port = df['port rets']
    
    port.name ='Pair{}'.format(i+1)
          
    port_ret = pd.concat([port_ret,port],axis=1)
    
    port2 = df2['port rets']
    
    port2.name ='Pair{}'.format(i+1)
          
    port_ret2 = pd.concat([port_ret2,port2],axis=1)
    
    port3 = df3['port rets']
    
    port3.name ='Pair{}'.format(i+1)
          
    port_ret3 = pd.concat([port_ret3,port3],axis=1)
    
    print("Pair{}".format(i+1))
    print("Sharpe Ratio : " , round(sharpe,2))
    print("CAGR : " , round(CAGR,2)*100,"%")
    print("Sharpe Ratio SL : " , round(sharpe_SL,2))
    print("CAGR SL : " , round(CAGR_SL,2)*100,"%")
    print("="*50)

#%%
port_total_ret = port_ret.sum(axis=1)/num_Pairs
port_total_ret_SL = port_ret_with_stoploss.sum(axis=1)/num_Pairs

Portfolio_TotalReturn = port_total_ret.cumsum()
Portfolio_Total_CumReturn = Portfolio_TotalReturn+1

Portfolio_TotalReturn_SL = port_total_ret_SL.cumsum()
Portfolio_Total_CumReturn_SL = Portfolio_TotalReturn_SL+1

port_total_ret2 = port_ret2.sum(axis=1)/num_Pairs
port_total_ret_SL2 = port_ret_with_stoploss2.sum(axis=1)/num_Pairs

Portfolio_TotalReturn2 = port_total_ret2.cumsum()
Portfolio_Total_CumReturn2 = Portfolio_TotalReturn2+1

Portfolio_TotalReturn_SL2 = port_total_ret_SL2.cumsum()
Portfolio_Total_CumReturn_SL2 = Portfolio_TotalReturn_SL2+1

port_total_ret3 = port_ret3.sum(axis=1)/num_Pairs
port_total_ret_SL3 = port_ret_with_stoploss3.sum(axis=1)/num_Pairs

Portfolio_TotalReturn3 = port_total_ret3.cumsum()
Portfolio_Total_CumReturn3 = Portfolio_TotalReturn3+1

Portfolio_TotalReturn_SL3 = port_total_ret_SL3.cumsum()
Portfolio_Total_CumReturn_SL3 = Portfolio_TotalReturn_SL3+1

print("Portfolio total cumulative return : ", round(100*(Portfolio_Total_CumReturn[-1]-1),2),"%" )
print("Portfolio sharpe ratio : ",round((((port_total_ret3.mean()- 0.0235/252)/ port_total_ret3.std()) * np.sqrt(252)),2) )
print("Portfolio return skewness : ", round(port_total_ret.skew(),2))
print("="*50)
print("Portfolio total cumulative return : ", round(100*(Portfolio_Total_CumReturn_SL[-1]-1),2),"%" )
print("Portfolio sharpe ratio with stop loss: ",round((((port_total_ret_SL.mean()- 0.0235/252)/ port_total_ret_SL.std()) * np.sqrt(252)),2) )
print("Portfolio return skewness with stop loss: ", round(port_total_ret_SL.skew(),2))
         
   
#%% Without stop loss

perf = pd.concat([Portfolio_Total_CumReturn,Portfolio_Total_CumReturn2,Portfolio_Total_CumReturn3],axis=1)
perf.columns = ['criteria (0,8,0.2)','criteria (0,9,0.1)','criteria (0,7,0.3)']
perf_ = perf.calc_stats()
perf_.plot(title='Portfolio Total Cumulative return without stop loss')
plt.ylabel('Cumulative Return')

best = Portfolio_Total_CumReturn3.calc_stats()

best.display_monthly_returns()
best.plot_histogram(title ='Return histogram',color='DarkGreen',alpha=0.5)
port_ret.columns = ['Pair1','Pair2','Pair3','Pair4','Pair5']   
ax = port_ret.hist(figsize=(12, 5))   
port_ret.plot_corr_heatmap(title ='Return Correlation',cmap='GnBu')
print(best.display())

#%% With stop loss

perf_SL = pd.concat([Portfolio_Total_CumReturn_SL,Portfolio_Total_CumReturn_SL2,Portfolio_Total_CumReturn_SL3],axis=1)
perf_SL.columns = ['criteria (0,8,0.2)','criteria (0,9,0.1)','criteria (0,7,0.3)']
perf_SL_ = perf_SL.calc_stats()
perf_SL_.plot(title='Portfolio Total Cumulative return with stop loss {}%'.format(loss_cut*100))
plt.ylabel('Cumulative Return')

best_ = Portfolio_Total_CumReturn3.calc_stats()

best_.display_monthly_returns()
best_.plot_histogram(title ='Return histogram',color='DarkGreen',alpha=0.5)
port_ret_with_stoploss.columns = ['Pair1','Pair2','Pair3','Pair4','Pair5']   
ax = port_ret_with_stoploss.hist(figsize=(12, 5))   
port_ret_with_stoploss.plot_corr_heatmap(title ='Return Correlation',cmap='GnBu')
print(best_.display())    

#%%

perf_each = port_ret.cumsum()+1
perf_each_result = perf_each.calc_stats()
print(perf_each_result.display())

#%%
#ffn.core.calc_stats(perf_each).to_csv(sep=',', path ='C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/통계적차익거래기법/perf_each.csv') #ffn data 저장
#ffn.core.calc_stats(perf).to_csv(sep=',', path ='C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/통계적차익거래기법/perf.csv') #ffn data 저장
#
