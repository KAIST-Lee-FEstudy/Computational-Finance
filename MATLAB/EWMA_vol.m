%%%%%%%% EWMA Volatility %%%%%%%%

function vol = EWMA_vol(price, lambda)

% �ְ��� �α׼��ͷ��� ��ȯ
logret = diff(log(price));

[m,n] =size(logret);

k = (1:m)';
h = (m:-1:1)';

% EWMA volatility ����
vol = zeros(1,n);
for i = 1:n
    vol(i) = sqrt((1-lambda)*sum(lambda.^(k-1).*logret(h,i).^2));
end
end

% ���� : price = xlsread('price.xlsx');
%        vol = EWMA_vol(price, 0.94)

    
