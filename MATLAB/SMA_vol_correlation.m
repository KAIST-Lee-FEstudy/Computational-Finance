% If you want to read data with table type 
% filename = 'index_price.xlsx';
% sheet = 'Sheet1';
% xlRange = 'A14:D1358';
% Indexprice = readtable(filename,'Sheet',sheet,'Range',xlRange);
% Indexprice.Properties.VariableNames={'Date','Kospi200','HSCEI','SPX500'};

%% Process
% 일별 데이터 로딩
Indexprice = xlsread('price.xlsx');

% 주가를 로그수익률로 전환
logret = diff(log(Indexprice));      

% SMA(Simple Moving Average) 추정
SMA_vol = std(logret)

% SMA Correlation 추정
SMA_corr = corr(logret)




