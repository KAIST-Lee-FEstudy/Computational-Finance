%%%%%%%% EWMA Volatility & Correlation %%%%%%%%

function [vol,rho] = EWMA_vol_correlation(price,lambda) 

% 일별 데이터 로딩
% price = xlsread(price);
% 주가를 로그수익률로 전환
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

% EWMA Correlation Matric 추정
rho = zeros(n);
for i=1:n
    for j =1:n
        rho(i,j) = cvar(i,j)/sqrt(cvar(i,i)*cvar(j,j));
    end
end

% EWMA Volatility 추정 
vol = sqrt(diag(cvar))'; % Varianc-Covariance Matrix의 대각원소 추출 

end

%실행 : price = xlsread('price.xlsx');
%       [vol,rho]=EWMA_vol_correlation(price,0.94) (월별 변동성 추정시 0.97 사용)






