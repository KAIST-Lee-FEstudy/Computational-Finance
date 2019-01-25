%%%%%%%%%%% EWMA log-likelihood function %%%%%%%%%%%

function f = EWMA_llh(b,x)
 
V = zeros(length(x)-1,1);
V(1) = x(1)^2;

for i = 2:length(x)-1
    V(i) = (1-b)*x(i)^2+b*V(i-1);   % b = lambda
end

f = sum(log(V)+(x(2:end).^2)./V);   % 로그우도함수
