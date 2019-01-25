%%%%%%%% EWMA Volatility & Correlation %%%%%%%%

function [vol,rho] = EWMA_vol_correlation(price,lambda) 

% �Ϻ� ������ �ε�
% price = xlsread(price);
% �ְ��� �α׼��ͷ��� ��ȯ
logret = diff(log(price)); 

[m,n] = size(logret);

k = (1:m)';
h = (m:-1:1)';

% EWMA Variance-Covariance 

cvar = zeros(n);
for i=1:n
    for j = 1:n
        cvar(i,j) = (1-lambda)*sum(lambda.^(k-1).*logret(h,i).*logret(h,j));
    end
end

% EWMA Correlation Matric ����
rho = zeros(n);
for i=1:n
    for j =1:n
        rho(i,j) = cvar(i,j)/sqrt(cvar(i,i)*cvar(j,j));
    end
end

% EWMA Volatility ���� 
vol = sqrt(diag(cvar))'; % Varianc-Covariance Matrix�� �밢���� ���� 

end

%���� : price = xlsread('price.xlsx');
%       [vol,rho]=EWMA_vol_correlation(price,0.94) (���� ������ ������ 0.97 ���)






