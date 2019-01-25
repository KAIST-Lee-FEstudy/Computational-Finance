%%%%%%%% European Vanilla option pricing (BS closed-form) %%%%%%%%

function price = BS_Vanilla(C_P,S,X,r,T,vol,q)

% 배당수익률 q를 입력하지 않을 경우 그냥 0으로 처리,q는 optional
if nargin < 7,q = 0;
end

% d1, d2 계산
d1 = (log(S/X)+(r-q+vol^2/2)*T)/(vol*sqrt(T));
d2 = d1 - vol*sqrt(T);

% price 계산
if C_P =='C'                % Call, Put flag
   % N(d1) : Normal cumulative probability density function , mean =0,std=1
   % normcdf 함수 이용
   price = S*exp(-q*T).*normcdf(d1)-X*exp(-r*T)*normcdf(d2); % call price
elseif C_P == 'P'
    price = -S*exp(-q*T).*normcdf(-d1)+X*exp(-r*T)*normcdf(-d2); % put price
end
end



