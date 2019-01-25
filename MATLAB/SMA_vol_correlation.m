% If you want to read data with table type 
% filename = 'index_price.xlsx';
% sheet = 'Sheet1';
% xlRange = 'A14:D1358';
% Indexprice = readtable(filename,'Sheet',sheet,'Range',xlRange);
% Indexprice.Properties.VariableNames={'Date','Kospi200','HSCEI','SPX500'};

%% Process
% �Ϻ� ������ �ε�
Indexprice = xlsread('price.xlsx');

% �ְ��� �α׼��ͷ��� ��ȯ
logret = diff(log(Indexprice));      

% SMA(Simple Moving Average) ����
SMA_vol = std(logret)

% SMA Correlation ����
SMA_corr = corr(logret)




