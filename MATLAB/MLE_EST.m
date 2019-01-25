%%%%%%%%%%% MLE SMA fitting %%%%%%%%%%%
Indexprice = xlsread('price.xlsx');
kospi200 = Indexprice(:,1);
% hsc = Indexprice(:,2);
%spx500 = Indexprice(:,3);
logret = diff(log(kospi200));
%logret = price2ret(spx500);
o = optimset('maxiter',500);

% MLE SMA
SMA_b0 = 0.01;                      % ������ ���(�л�)�� �ʱⰪ
% �α׿쵵�Լ� �ִ�ȭ
[SMA_var, SMA_fval] = fminsearch(@SMA_llh,SMA_b0,o,logret)

% MLE EWMA
EWMA_b0 = 0.7;                  % ������ ���(lambda)�� �ʱⰪ
% �α׿쵵�Լ� �ִ�ȭ
[EWMA_para,EWMA_fval] = fminsearch(@EWMA_llh,EWMA_b0,o,logret)

% MLE GARCH
% ������ ��� (alpha,beta)�� �ʱⰪ(�뷫 0<alpha<0.25,beta>0.7,alpha+beta<1���� ����)
GARCH_b0 = [0.1,0.8];
%�α׿쵵�Լ� �ִ�ȭ
[GARCH_para,GARCH_fval] = fminsearch(@GARCH_llh,GARCH_b0,o,logret)
