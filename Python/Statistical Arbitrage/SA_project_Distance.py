# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 02:40:53 2018

@author: rbgud
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller

#directory
path = os.getcwd()

#the file with closing prices of all the stocks should be in the same directory as this notebook
prices = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/통계적차익거래기법/close_new.csv',engine='python')
data = pd.DataFrame()

#setting index
prices['Date'] = pd.to_datetime(prices['Date'])
prices.set_index('Date',inplace = True)

#save a dataframe "data" as the normalized prices
data = prices/prices.iloc[0]

#dividing based on formation period and trading period
#trading_data is the price data for the trading period
trading_data = data.loc[data.index > '2017-09-08']

#save data as the data for the formation period
data = data.loc[data.index < '2017-09-09']

#save the trading data pric
prices = prices.loc[prices.index > '2017-09-09']

#trade cost of 30bps
trade_cost = 0.003

#list of pairs
selected_pairs = [['HON','NEE'], ['TXN','SYK'],['BDX','SYK'], ['HON','DHR'],['JPM','PNC']]
selected_pairs2 = [['HON','MMC'], ['HON','AON'],['BBT','STI'], ['DUK','SRE'],['LMT','NOC']]
half_life_sort = [['HON','NEE'], ['CMCSA','HON'],['MRK','SRE'], ['UPS','D'],['CCL','RHT']]
pearson_sort = [['MMM','MMC'], ['BDX','SYK'],['INTU','MMC'], ['MMC','APH'],['JPM','PNC']]
SSD_pairs = [['PG','DUK'], ['D','UPS'],['USB','BBT'], ['T','SO'],['WMT','LIN']]

#%%

#tfunction takes a pair of ticker symbols as strings
#outputs a list of trades based on the data from the trading period

def trading_signals(first, second, trading_data = trading_data, formation_data = data):
    #choose 2-sigma as the trading signal
    signal = 2*np.std(formation_data[first] - formation_data[second])
    result_dict = {}
    
    #there should be no trading initially
    trading = False
    
    #create a time series of the spread between the two stocks
    differences = trading_data[first] - trading_data[second]
    for i in range(len(differences)):
        
        #if there is no trading, OPEN it if the spread is greater than the signal
        #AND the spread is less than the stop-loss of 4-sigma
        #if not, move onto the next day
        if trading == False:
            if abs(differences.iloc[i]) > signal and abs(differences.iloc[i] < 2*signal):
                trading = True
                start_date = differences.index.values[i]
                
        #if the trade is already open, we check to see if the spread has crossed OR exceeded the 4-sigma stoploss
        #we close the trade and record the start and end date of the trade
        #we also record the return from the short and long position of the trade
        else:
            if (differences.iloc[i-1] * differences.iloc[i] < 0) or (i == len(differences)-1) or abs(differences.iloc[i] > 2*signal):
                trading = False
                end_date = differences.index.values[i]
                if differences[i-1] > 0:
                    s_ret = (trading_data[first][start_date] - trading_data[first][end_date])/trading_data[first][start_date]
                    l_ret = (trading_data[second][end_date] - trading_data[second][start_date])/trading_data[second][start_date]
                    result_dict[start_date] = [first, second, start_date, end_date, s_ret,l_ret]
                else:
                    s_ret = (trading_data[second][start_date] - trading_data[second][end_date])/trading_data[second][start_date]
                    l_ret = (trading_data[first][end_date] - trading_data[first][start_date])/trading_data[first][start_date]
                    result_dict[start_date] = [second, first, start_date, end_date, s_ret,l_ret]
    
    #formatting the final dataframe to be returned
    df = pd.DataFrame.from_dict(result_dict, orient = 'index', columns = ['Short','Long','Start','End', 'SReturn','LReturn'])
    df.index = list(range(len(df)))
    df['Total'] = df['SReturn'] + df['LReturn']
    df['Length'] = (df['End'] - df['Start']).dt.days
    return (df, len(df))

#%%
#this function takes as its input a dataframe returned from the trading_signals function above
#it takes the signals and builds a day by day portfolio based on the signals

def build_portfolio(trade_list, trading_data = trading_data):
    #create a index_list of dates
    index_list = trading_data.index.tolist()
    
    #initialize dataframe
    portfolio = pd.DataFrame(index = trading_data.index.values, columns = ['Short','Long','ShortR','LongR','Trading'])
    l = trade_list[1]
    trade_list = trade_list[0]
    
    #for each trade, find the start and end dates, and which stocks to long/short
    for i in range(len(trade_list)):
        start = trade_list['Start'][i]
        end = trade_list['End'][i]
        short = trade_list['Short'][i]
        lon = trade_list['Long'][i]
        di = index_list.index(start)
        di2 = index_list.index(end)
        
        #from the start to end date, add the value of the position from that day for that stock
        #also take away trade cost (for long) or add it for shorts
        for j in range(di2 - di + 1):
            date_index = di + j
            dt = index_list[date_index]
            portfolio['Short'][dt] = (trading_data[short][dt]/trading_data[short][index_list[di]]) + trade_cost
            portfolio['Long'][dt] = trading_data[lon][dt]/trading_data[lon][index_list[di]] - trade_cost
            portfolio['Trading'][dt] = 1
            if j == (di2 - di):
                portfolio['Short'][dt] = portfolio['Short'][dt] + trade_cost
                portfolio['Long'][dt] = portfolio['Long'][dt] - trade_cost

    #fill non-trading days
    portfolio.fillna(value = 0, axis = 0)
    
    #adding columns for returns from the short and long portions of the portfolio
    for j in range(1, len(portfolio)):
        if portfolio.iloc[j-1]['Short'] > 0:
            portfolio.iloc[j]['ShortR'] = -(portfolio.iloc[j]['Short'] - portfolio.iloc[j-1]['Short'])/portfolio.iloc[j-1]['Long']
            portfolio.iloc[j]['LongR'] = (portfolio.iloc[j]['Long'] - portfolio.iloc[j-1]['Long'])/portfolio.iloc[j-1]['Long']
        else:
            portfolio.iloc[j]['ShortR'] = 0
            portfolio.iloc[j]['LongR']= 0
            
    #total return is teh sum of both returns
    portfolio['Total'] = portfolio['ShortR'] + portfolio['LongR']
    portfolio.fillna(0, inplace = True)
    return (portfolio, l)    

#%%

#this function is a rapper function that takes in pairs and passes them along to the trading_signals function
#this result is then passed onto the build_portfolio function
#if there are multiple pairs it adds up the results of all the portfolios generated for each pair
#the overall portfolio performance is then analyzed
#the result is a list with [anualized return, SD, sharpe ratio, and # of trades]

def analyze_portfolio(pairs):
    i = 0
    df = (build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[0])
    trade_count = build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
    for i in range(1, len(pairs)):
        df = df + (build_portfolio(trading_signals(pairs[i][0], pairs[i][1])))[0]
        trade_count += build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
    df_short = df['ShortR']/5
    df_long = df['LongR']/5
    df_final = pd.concat([df_short, df_long], axis=1)
    df_final.columns = ['Short Return','Long Return']
    df_final.index.name = 'Date'
    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
    df_final.fillna(0, inplace = True)
    arithemtic_daily_mean = np.mean(df_final['Total'])
    annualized_return = (1+arithemtic_daily_mean)**250 - 1
    annualized_std = np.std(df_final['Total'])*np.sqrt(250)
    sharpe_ratio = annualized_return/annualized_std
    return [annualized_return, annualized_std, sharpe_ratio, trade_count]

#%%
#create different portfolios based on the set of pairs we selected above

optimal_selection = (analyze_portfolio(selected_pairs))
pvalue_selection = (analyze_portfolio(half_life_sort))
pearson_selection = (analyze_portfolio(pearson_sort))
SSD_selection = (analyze_portfolio(SSD_pairs))
selected_pairs2 = (analyze_portfolio(selected_pairs2))
print('Done')    
#%%
#show and compare the results of the portfolios from above

comparison_df = pd.DataFrame(data = np.array([optimal_selection, pvalue_selection, pearson_selection, SSD_selection, selected_pairs2]))
comparison_df.index = np.array(['Industry Matched', 'Sorted by p-value','Sorted by Pearson Correlation', 'Sorted by SSD', 'ML'])
comparison_df.columns = ['Annualized Mean Return','Annualized SD of Daily Returns','Sharpe Ratio','Total Trades']
comparison_df['Annualized Mean Return'] = pd.Series(["{0:.2f}%".format(val * 100) for val in comparison_df['Annualized Mean Return']], index = comparison_df.index)
comparison_df['Annualized SD of Daily Returns'] = pd.Series(["{0:.2f}%".format(val * 100) for val in comparison_df['Annualized SD of Daily Returns']], index = comparison_df.index)
comparison_df['Total Trades'] = (comparison_df['Total Trades']*4)
comparison_df

