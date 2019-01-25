%%%%%%%%%%% GARCH(1,1) log-likelihood function %%%%%%%%%%%

function f = GARCH_llh(b,x)   % b는 추정할 모수, x는 실제 발생 수익률 

VL = var(x);                  % variance targeting, 표본분산을 장기평균분산으로 대체하여 추정할 모수 개수 줄임 
V = zeros(length(x)-1,1);
V(1) = x(1)^2;                % 첫 번째 분산 예측값을 첫 번째 수익률의 제곱으로 설정 

for i = 2:length(x)-1
    % sigma(t)^2 = w + alpha*x(t-1)^2+beta*sigma(t-1)^2
    V(i) = (1-b(1)-b(2))*VL+b(1)*x(i)^2+b(2)*V(i-1);
end
%sigmat = V(length(x)-1)
%f = sum(log(V)+(x(2:end).^2)./V); % 우도함수
   
end