%%%%%%%%%%% MLE SMA fitting %%%%%%%%%%%
Indexprice = xlsread('price.xlsx');
kospi200 = Indexprice(:,1);
% hsc = Indexprice(:,2);
%spx500 = Indexprice(:,3);
logret = diff(log(kospi200));
%logret = price2ret(spx500);
o = optimset('maxiter',500);

% MLE SMA
SMA_b0 = 0.01;                      % 추정할 모수(분산)의 초기값
% 로그우도함수 최대화
[SMA_var, SMA_fval] = fminsearch(@SMA_llh,SMA_b0,o,logret)

% MLE EWMA
EWMA_b0 = 0.7;                  % 추정할 모수(lambda)의 초기값
% 로그우도함수 최대화
[EWMA_para,EWMA_fval] = fminsearch(@EWMA_llh,EWMA_b0,o,logret)

% MLE GARCH
% 추정할 모수 (alpha,beta)의 초기값(대략 0<alpha<0.25,beta>0.7,alpha+beta<1에서 설정)
GARCH_b0 = [0.1,0.8];
%로그우도함수 최대화
[GARCH_para,GARCH_fval] = fminsearch(@GARCH_llh,GARCH_b0,o,logret)
