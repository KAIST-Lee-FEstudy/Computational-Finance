%%%%%%%% EWMA Volatility %%%%%%%%

function vol = EWMA_vol(price, lambda)

% 주가를 로그수익률로 전환
logret = diff(log(price));

[m,n] =size(logret);

k = (1:m)';
h = (m:-1:1)';

% EWMA volatility 추정
vol = zeros(1,n);
for i = 1:n
    vol(i) = sqrt((1-lambda)*sum(lambda.^(k-1).*logret(h,i).^2));
end
end

% 실행 : price = xlsread('price.xlsx');
%        vol = EWMA_vol(price, 0.94)

    
