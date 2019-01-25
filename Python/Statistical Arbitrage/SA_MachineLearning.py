# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:49:48 2018

@author: rbgud
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import scale
import statsmodels.api as sm
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#%% Get Data

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

returns_copy=returns
returns=returns.iloc[:int(len(returns)*0.8),:]  #Formation Period 동안의 수익률로 페어 찾고 분석 
returns=returns.dropna(axis=1)
#returns=returns.fillna(1)


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
''' PCA , DBSCAN으로 클러스터링 (후보군 찾기) 
    70%정도의 설명력을 갖는 40개 components를 활용하여 분석
'''

N_PRIN_COMPONENTS = 40
pca = PCA(n_components=N_PRIN_COMPONENTS)

ttt = pca.fit_transform(returns)

tt = pca.components_
loading = pca.components_.T*np.sqrt(pca.explained_variance_)
loading = StandardScaler().fit_transform(loading)

#pca.components_.T.shape

X=np.array(loading)

# Density-Based Spatial Clustering of Applications with Noise

clf = DBSCAN(eps=1.8, min_samples=3)
clf.fit(X)
labels = clf.labels_                                          # -1이면 구분이 안된것
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   # 숫자 5, 총 5개의 클러스터 있음

print ("\nClusters discovered: %d" % n_clusters_)

clustered = clf.labels_   #{-1,-1,.....0....4... } 총 200개 사이즈

#ticker_count = len(returns.columns) 

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

#%% 차원 축소하여 Cluster plot을 출

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
    cmap=cm.Paired
)

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

def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
        Close[tickers]
    )
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs


pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])



print ("In those pairs, there are %d unique tickers.\n" % len(np.unique(pairs)))
print(pairs)


stocks = np.unique(pairs)
X_df = pd.DataFrame(index=returns.T.index, data=X)
in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.loc[stocks]


#%%
''' 페어 저장 '''
# ML Base Pairs
pairs = [('HON UN Equity', 'MMC UN Equity'), ('HON UN Equity', 'AON UN Equity'), ('BBT UN Equity', 'STI UN Equity'), ('DUK UN Equity', 'SRE UN Equity'), ('LMT UN Equity', 'NOC UN Equity')]

# Industry Matched Pairs
#pairs = [('HON UN Equity','NEE UN Equity'), ('TXN UW Equity','SYK UN Equity'),('BDX UN Equity','SYK UN Equity'), ('HON UN Equity','DHR UN Equity'),('JPM UN Equity','PNC UN Equity')]

#SSD Pairs
#pairs = [('PG UN Equity','DUK UN Equity'), ('D UN Equity','UPS UN Equity'),('USB UN Equity','BBT UN Equity'), ('T UN Equity','SO UN Equity'),('WMT UN Equity','LIN UN Equity')]

#Pearson Pairs
#pairs = [('MMM UN Equity','MMC UN Equity'), ('BDX UN Equity','SYK UN Equity'),('INTU UW Equity','MMC UN Equity'), ('MMC UN Equity','APH UN Equity'),('JPM UN Equity','PNC UN Equity')]

#Half-Life Pairs
#pairs = [('HON UN Equity','NEE UN Equity'), ('CMCSA UW Equity','HON UN Equity'),('MRK UN Equity','SRE UN Equity'), ('UPS UN Equity','D UN Equity'),('CCL UN Equity','RHT UN Equity')]

#pairs = [('PNC UN Equity', 'BBT UN Equity'), ('DUK UN Equity', 'SRE UN Equity'), ('HON UN Equity', 'DHR UN Equity'), ('CL UN Equity', 'KMB UN Equity'), ('BDX UN Equity', 'SYK UN Equity')]

#%%


'''찾은 페어들의 종가 그래프와 스프레드 Plot '''

for i in range(len(pairs)):
    
     
     Close[list(pairs[i])].plot()                                   
     plt.axvline(x='2017-09-08')
     ax = (Close[pairs[i][1]]-Close[pairs[i][0]]).plot(title='Stock price of pairs and spread')
     ax.legend(['{}'.format(pairs[i][0]),'{}'.format(pairs[i][1]),'Spread'])

#%% Spread 모델을 이용하기 위한 회귀분석 및 Support Vector Machine 학습을 위한 T-score 계산

# Spread Model 구성을 위해 data 전처리 
model_close=Close
model_open=Open

model_wma_close=Wma
model_wma_open=model_wma_close.shift(1)

model_sma5_close=Sma5
model_sma5_open=model_sma5_close.shift(1)

model_sma14_close=Sma14
model_sma14_open=model_sma14_close.shift(1)

model_rsi_close=Rsi
model_rsi_open=model_rsi_close.shift(1)

model_mfi5_close=Mfi5
model_mfi5_open=model_mfi5_close.shift(1)  

model_mfi14_close=Mfi14
model_mfi14_open=model_mfi14_close.shift(1)

model_volume_close = volume
model_volume_open=model_volume_close.shift(1)  
     
''' Spread 모델을 이용하기 위한 회귀분석 및 T-score 계산'''
def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    results = sm.OLS(y, X).fit()
    return results

mod = sys.modules[__name__]
     
def Caculate_Tscore(case):
      
    if case == 1:                        
    
       data1_close = model_close
       data2_open = model_open 
    
    elif case == 2:                       
         
         data1_close = model_wma_close
         data2_open = model_wma_open 
        
    elif case == 3:                       
         data1_close = model_sma5_close
         data2_open = model_sma5_open 
         
    elif case == 4:                       
         data1_close = model_rsi_close
         data2_open = model_rsi_open    
         
    elif case == 5:                       
         data1_close = model_mfi5_close 
         data2_open = model_mfi5_open
         
    elif case == 6:                       
         data1_close = model_mfi14_close
         data2_open = model_mfi14_open  
         
    elif case == 7:                       
         data1_close = model_sma14_close
         data2_open = model_sma14_open      
         
    elif case == 8:                       
         data1_close = model_volume_close
         data2_open = model_volume_open      
         
   
    for i in range(len(pairs)):
        
        d_A = data1_close[pairs[i][0]]-data2_open[pairs[i][0]]
        d_B= data1_close[pairs[i][1]]-data2_open[pairs[i][1]]   
        
        d_A = d_A.dropna()
        d_B = d_B.dropna()
        
        if case == 1: 
                
           Y= (d_A/data2_open[pairs[i][0]])
           X= (d_B/data2_open[pairs[i][1]])
            
        else:
            
            Y= (d_A/data2_open[pairs[i][0]]).dropna()
            X= (d_B/data2_open[pairs[i][1]]).dropna()
             
        result=reg_m(Y,X)
        X_t=result.resid
        
        resid=result.resid
        resid_1=resid.shift(1).dropna()
        resid=resid.iloc[1:]
        result=reg_m(resid, resid_1)
        
        const=result.params[1]
        beta=result.params[0]
        error=result.resid
        
        mu=const/(1-beta)
        sigma= np.sqrt( error.var()/(1-(beta**2)) )
        
        if case == 1:
            
           setattr(mod, 'T_Price_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] )
                   
           buy_open= (pd.DataFrame(X_t > 0) & pd.DataFrame(X_t.shift(-1) < 0.8*X_t))*1
           sell_open= (pd.DataFrame(X_t < 0) & pd.DataFrame(X_t.shift(-1) > 0.8*X_t))*-1
                                 
           for j in range(100):
                
                buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1)*pd.DataFrame(X_t > 0)*1
                          
                sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1)* pd.DataFrame(X_t<0)*-1 
                   
           setattr(mod, 'label_{}'.format(i), pd.DataFrame( buy_open+sell_open,dtype='i').iloc[1:] ) 
                       
       
        elif case == 2:
            
           setattr(mod, 'T_wma_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ))
           
        elif case == 3:   
          
           setattr(mod, 'T_sma5_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ))
           
        elif case == 4:
            
           setattr(mod, 'T_rsi_{}'.format(i), pd.DataFrame( (X_t - mu) /sigma  ))
        
        elif case == 5:
            
           setattr(mod, 'T_mfi5_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ) )
           
        elif case == 6:
        
           setattr(mod, 'T_mfi14_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ))
           
        elif case == 7:   
          
           setattr(mod, 'T_sma14_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ))
           
        elif case == 8:   
          
           setattr(mod, 'T_volume_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma  ) )    
       
          
    return 

#%% T-score data set 출력
        
for i in range(1,9):

    Caculate_Tscore(i)
    
      
label_0 = pd.DataFrame(np.array(label_0,dtype='int')).set_index(T_Price_0.index)
label_1 = pd.DataFrame(np.array(label_1,dtype='int')).set_index(T_Price_0.index)
label_2 = pd.DataFrame(np.array(label_2,dtype='int')).set_index(T_Price_0.index)
label_3 = pd.DataFrame(np.array(label_3,dtype='int')).set_index(T_Price_0.index)
label_4 = pd.DataFrame(np.array(label_4,dtype='int')).set_index(T_Price_0.index)

#%% Concate T-score data set for trainning SVM Model case2

Pair1 = pd.concat([T_Price_0,T_mfi14_0,T_rsi_0,T_sma5_0,T_wma_0,label_0],axis=1,join='inner').values

Pair2 = pd.concat([T_Price_1,T_mfi14_1,T_rsi_1,T_sma5_1,T_wma_1,label_1],axis=1,join='inner').values

Pair3 = pd.concat([T_Price_2,T_mfi14_2,T_rsi_2,T_sma5_2,T_wma_2,label_2],axis=1,join='inner').values

Pair4 = pd.concat([T_Price_3,T_mfi14_3,T_rsi_3,T_sma5_3,T_wma_3,label_3],axis=1,join='inner').values

Pair5 = pd.concat([T_Price_4,T_mfi14_4,T_rsi_4,T_sma5_4,T_wma_4,label_4],axis=1,join='inner').values
   

#%% Make Train-Test Data set

def MinMaxScale(data):
   
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    
    scale_data = numerator / (denominator + 1e-7)
    
    return scale_data

def makeX_train_test(data,rate):
    
    data = MinMaxScale(data)
    
    X_train = data[:int(len(data)*rate),:np.shape(data)[1]-1]
    X_test =  data[int(len(data)*rate):,:np.shape(data)[1]-1]
    
    return X_train, X_test

def makeY_train_test(data,rate):
     
    Y_train = data[:int(len(data)*rate),-1]
    Y_test = data[int(len(data)*rate):,-1]
    
    return Y_train, Y_test


#%%
    
rate = 0.8

Pair1X_train,Pair1X_test = makeX_train_test(Pair1,rate)
Pair1Y_train,Pair1Y_test = makeY_train_test(Pair1,rate)  

Pair2X_train,Pair2X_test = makeX_train_test(Pair2,rate)
Pair2Y_train,Pair2Y_test = makeY_train_test(Pair2,rate)
    
Pair3X_train,Pair3X_test = makeX_train_test(Pair3,rate)
Pair3Y_train,Pair3Y_test = makeY_train_test(Pair3,rate)  

Pair4X_train,Pair4X_test = makeX_train_test(Pair4,rate)
Pair4Y_train,Pair4Y_test = makeY_train_test(Pair4,rate)  

Pair5X_train,Pair5X_test = makeX_train_test(Pair5,rate)
Pair5Y_train,Pair5Y_test = makeY_train_test(Pair5,rate)  


#%% Train Model for Pair

def Train_SVM(Xtrain,Xtest,Ytrain,Ytest, C, gamma,decision_function,kernel):
    
    clf = svm.SVC(C=c,gamma=gamma, decision_function_shape = decision_function,kernel = kernel)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
      
    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100)) 
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
  
    return pred_df

def Train_LinearSVM(Xtrain,Xtest,Ytrain,Ytest, C,multi_class,tol,max_iter):
    
    clf = svm.LinearSVC(C=C,multi_class=multi_class,tol=tol,max_iter=max_iter)
    
    clf.fit(Xtrain,Ytrain)
    
    y_pred = clf.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred) 

    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))

    return pred_df     

def Train_GradientBoostingCalssifier(Xtrain,Xtest,Ytrain,Ytest,learning_rate,n_estimators):
    
    model = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)
    
    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
    
    return pred_df

def Train_RandomForestClassifier(Xtrain,Xtest,Ytrain,Ytest,n_estimators,criterion):
    
    model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)

    model.fit(Xtrain, Ytrain)
    
    y_pred = model.predict(Xtest)
    
    pred_df = pd.DataFrame(y_pred)
    
    conf_matrix=confusion_matrix(Ytest, y_pred)

    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    print(conf_matrix)
    print(classification_report(Ytest, y_pred))
   
    return pred_df
    

#%% SVM Train Result

# Hyper parameters for SVM & LinearSVM
c = 15
gamma = 0.5
decision_function ='ovr'
kernel = 'linear'
multi_class = 'ovr'
tol = 0.0001
max_iter = 10000
#
# Hyper parameters for GradientBoostingClassifier
learning_rate_gb = 1
n_estimators_GB = 100

# Hyper parameters for GradientBoostingClassifier
n_estimators_RF = 30
criterion = 'entropy'

# Train result

print('-------------------------------------------------------------------------')
print('------------------------------Result of SVM------------------------------')
print('-------------------------------------------------------------------------')   
pred1_SVM = Train_SVM(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred2_SVM = Train_SVM(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred3_SVM = Train_SVM(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred4_SVM = Train_SVM(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred5_SVM = Train_SVM(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,c,gamma,decision_function,kernel)

print('-------------------------------------------------------------------------')
print('---------------------------Result of LinearSVM---------------------------')
print('-------------------------------------------------------------------------')
pred1_LSVM = Train_LinearSVM(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test, c,multi_class,tol,max_iter)
print('-------------------------------------------------------------------------')
pred2_LSVM = Train_LinearSVM(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test, c,multi_class,tol,max_iter)
print('-------------------------------------------------------------------------')
pred3_LSVM = Train_LinearSVM(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test, c,multi_class,tol,max_iter)
print('-------------------------------------------------------------------------')
pred4_LSVM = Train_LinearSVM(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test, c,multi_class,tol,max_iter)
print('-------------------------------------------------------------------------')
pred5_LSVM = Train_LinearSVM(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test, c,multi_class,tol,max_iter)

print('-------------------------------------------------------------------------')
print('------------------------------Result of GBC------------------------------')
print('-------------------------------------------------------------------------')
pred1_GBC = Train_GradientBoostingCalssifier(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred2_GBC = Train_GradientBoostingCalssifier(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,learning_rate_gb,n_estimators_GB)  
print('-------------------------------------------------------------------------')
pred3_GBC = Train_GradientBoostingCalssifier(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred4_GBC = Train_GradientBoostingCalssifier(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred5_GBC = Train_GradientBoostingCalssifier(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,learning_rate_gb,n_estimators_GB)

print('-------------------------------------------------------------------------')
print('------------------------------Result of RF------------------------------')
print('-------------------------------------------------------------------------')
pred1_RF = Train_RandomForestClassifier(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred2_RF = Train_RandomForestClassifier(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,n_estimators_RF,criterion)  
print('-------------------------------------------------------------------------')
pred3_RF = Train_RandomForestClassifier(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred4_RF = Train_RandomForestClassifier(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred5_RF = Train_RandomForestClassifier(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,n_estimators_RF,criterion)



#%% Prediction Result

Pair1_pred = pd.concat([pred1_SVM,pred1_LSVM,pred1_GBC,pred1_RF],axis=1)

Pair1_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair2_pred = pd.concat([pred2_SVM,pred2_LSVM,pred2_GBC,pred2_RF],axis=1)

Pair2_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair3_pred = pd.concat([pred3_SVM,pred3_LSVM,pred3_GBC,pred3_RF],axis=1)

Pair3_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair4_pred = pd.concat([pred4_SVM,pred4_LSVM,pred4_GBC,pred4_RF],axis=1)

Pair4_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair5_pred = pd.concat([pred5_SVM,pred5_LSVM,pred5_GBC,pred5_RF],axis=1)

Pair5_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']


#%%

def get_voting_score(data):
    
    data['Final_prediction'] = data['pred1_SVM']+data['pred1_LSVM']+\
                                     data['pred1_GBC']+data['pred1_RF']
    
    for i in range(len(Pair1_pred)):
                                
        if data['Final_prediction'][i]== 2 or data['Final_prediction'][i]== -2:
        
           data['Final_prediction'][i]= data['Final_prediction'][i]/2   
        
        elif data['Final_prediction'][i]== 3 or data['Final_prediction'][i]== -3:                               
        
             data['Final_prediction'][i]= data['Final_prediction'][i]/3 
             
        elif data['Final_prediction'][i]== 4 or data['Final_prediction'][i]== -4:
        
             data['Final_prediction'][i]= data['Final_prediction'][i]/4  
             
        else:
            
             data['Final_prediction'][i] =0
         
    df = data.values[:,-1]        

    return df

Pair1_pred_vote = get_voting_score(Pair1_pred)
Pair2_pred_vote = get_voting_score(Pair2_pred)
Pair3_pred_vote = get_voting_score(Pair3_pred)
Pair4_pred_vote = get_voting_score(Pair4_pred)
Pair5_pred_vote = get_voting_score(Pair5_pred)

Pairs_pred_vote = [Pair1_pred_vote,Pair2_pred_vote,Pair3_pred_vote,Pair4_pred_vote,Pair5_pred_vote]

reshape_len = len(Pair1X_test)
#
#Pairs_pred_vote = [pred1_RF.values.reshape((reshape_len)),pred2_RF.values.reshape((reshape_len)),pred3_RF.values.reshape((reshape_len))
#                    ,pred4_RF.values.reshape((reshape_len)),pred5_RF.values.reshape((reshape_len))]

#Pairs_pred_vote = [pred1_SVM.values.reshape((reshape_len)),pred2_SVM.values.reshape((reshape_len)),pred3_SVM.values.reshape((reshape_len))
#                    ,pred4_SVM.values.reshape((reshape_len)),pred5_SVM.values.reshape((reshape_len))]

Pairs_pred_vote = [pred1_LSVM.values.reshape((reshape_len)),pred2_LSVM.values.reshape((reshape_len)),pred3_LSVM.values.reshape((reshape_len))
                    ,pred4_LSVM.values.reshape((reshape_len)),pred5_LSVM.values.reshape((reshape_len))]

def get_cumret(data):
    
    ret = -1*data*(Close[pairs[i][0]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])+\
             data*(Close[pairs[i][1]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])
             
    cumret = pd.DataFrame(ret).cumsum()+1        
    
    return cumret,ret


#%%
    
Final_payoff = pd.DataFrame()

pair_ret_list = []
ret_each = pd.DataFrame()

plt.figure(figsize = (12,7)) 
for i in range(len(pairs)):
    
 
    cumret_= get_cumret(Pairs_pred_vote[i])[0]
    ret_ = get_cumret(Pairs_pred_vote[i])[1]
    
    ret_each = pd.concat([ret_each,ret_],axis=1)
    
    Final_payoff = pd.concat([Final_payoff,cumret_],axis=1)
    
    
    plt.plot(Final_payoff.index,cumret_,label = 'Pair{}'.format(i+1))

     
plt.xlabel('Date')
plt.ylabel('Cumulative Return(%)')
plt.title('Cumulative Return of pairs without stop loss')   
plt.legend(loc = 'upper left') 
plt.show()
ret_each.columns = ['0','1','2','3','4']

ret_sig = pd.DataFrame()
total_ = pd.DataFrame()
ret_sig_list=[]

for i in range(len(pairs)):
    
    ret_sig = pd.concat([ret_sig,ret_each['{}'.format(i)]],axis=1)
    
for i in range(len(pairs)):
    
    pred = pd.DataFrame(Pairs_pred_vote[i])
    total_ = pd.concat([total_,pred],axis=1) 
    
total = pd.concat([ret_sig.reset_index(drop=True),total_],axis=1)    

total = total.set_index(ret_each.index)
    
total.columns = [0,1,2,3,4,'pred1','pred2','pred3','pred4','pred5',]        

# loss cut

loss_cut = -0.1
    
for i in range(len(total.columns)-6):
    for j in range(len(total[i])-1):
        if i ==0:
            if ((total[i][j] < loss_cut) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total[i][j+1] = 0
                
        elif i ==1:
            if ((total[i][j] < loss_cut) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total[i][j+1] = 0        
        elif i ==2:
            if ((total[i][j] < loss_cut) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total[i][j+1] = 0
        elif i ==3:
            if ((total[i][j] < loss_cut) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total[i][j+1] = 0
        elif i ==4:
            if ((total[i][j] < loss_cut) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total[i][j+1] = 0

total.ix['2018-06-08'][3]=0
total['loss_cut']= (total[0]+total[1]+total[2]+total[3]+total[4])/5
total_cum = (total[[0,1,2,3,4]].cumsum(axis=0)+1)
total_cum.columns = ['Pair1','Pair2','Pair3','Pair4','Pair5']

plt.figure(figsize = (12,7))
plt.plot(total_cum.index,total_cum)
#total_cum.plot()
plt.xlabel('Date')
plt.ylabel('Cumulative Return(%)')
plt.title('Cumulative Return of pairs with stop loss')
plt.legend(['Pair1','Pair2','Pair3','Pair4','Pair5'],loc='upper left') 
plt.show()

plt.figure(figsize = (12,7))
((total['loss_cut']+1).cumprod()-1).plot() 
plt.xlabel('Date')
plt.ylabel('Cumulative Return(%)')
plt.title('Pairs Trading Portfolio Cumulative Return without cost')
plt.show()  


#
PF_Total_Return = pd.DataFrame()

for i in range(len(pairs)):
    
     ret_ = -1*Pairs_pred_vote[i]*(Close[pairs[i][0]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])+\
             Pairs_pred_vote[i]*(Close[pairs[i][1]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])
             
     PF_Total_Return = pd.concat([PF_Total_Return,ret_],axis=1)    
     
PF_Total_Return_sum = PF_Total_Return.sum(axis=1)/5


#
PF_True_Return = pd.DataFrame()

Pair_test_train = [Pair1Y_test,Pair2Y_test,Pair3Y_test,Pair4Y_test,Pair5Y_test]

for i in range(len(pairs)):
    
     ret_ = -1*Pair_test_train[i]*(Close[pairs[i][0]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])+\
             Pair_test_train[i]*(Close[pairs[i][1]].pct_change().shift(-1).loc[T_mfi14_0.index][int(len(T_mfi14_0)*rate):])
             
     PF_True_Return = pd.concat([PF_True_Return,ret_],axis=1)    

PF_True_Return_cum = PF_True_Return.cumsum()+1
#
cost = pd.DataFrame()

for i in range(len(pairs)):
    
    
    cost_ = (pd.DataFrame(Pairs_pred_vote[i]).replace(-1,1).sum() - 
        (pd.DataFrame(Pairs_pred_vote[i]).replace(0,np.nan) 
         == pd.DataFrame(Pairs_pred_vote[i]).replace(0,np.nan).shift(1)).sum())*0.003 
    
    cost = pd.concat([cost,cost_],axis=1)

cost_sum = cost.sum(axis=1)/5
PF_Total_Return_cumsum = PF_Total_Return_sum.cumsum()+1 
PF_Total_Return_cumsum[0] =1   

#plt.figure(figsize = (12,7))
#plt.plot(PF_Total_Return_cumsum.index,PF_Total_Return_cumsum)
#plt.show()

Total_Return = np.round((PF_Total_Return_cumsum[-1]-1)*100,2)
Total_Cost = np.round(cost_sum[0]*100,2)

print("Total Return : ", np.round(((total['loss_cut']+1).cumprod()-1).iloc[-2],3)*100,"%")
print("Total Cost : ", Total_Cost,"%")
print("Net Portfolio Return : ", np.round(Total_Return-Total_Cost,2),"%" )
print("Net Portfolio sharpe ratio : ",((((PF_Total_Return_sum.mean())/ PF_Total_Return_sum.std()) * np.sqrt(252)) ))
print("Net Portfolio return skewness : ", PF_Total_Return_sum.skew())
print("학습기간 누적 수익률 :\n", np.round(PF_True_Return_cum.iloc[-2,:]-1,2)*100)


#%%
pair_sharpe_stoploss = (total[[0,1,2,3,4]].mean(axis=0)/total[[0,1,2,3,4]].std(axis=0))*np.sqrt(252)

pair_skewness_stoploss = total[[0,1,2,3,4]].skew(axis=0)

pair_sharpe_nostoploss = (ret_each[['0','1','2','3','4']].mean(axis=0)/ret_each[['0','1','2','3','4']].std(axis=0))*np.sqrt(252)

pair_skewness_nostoploss = ret_each[['0','1','2','3','4']].skew(axis=0)

