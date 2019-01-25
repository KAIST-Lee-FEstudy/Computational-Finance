%%%%%%%%%%% GARCH(1,1) log-likelihood function %%%%%%%%%%%

function f = GARCH_llh(b,x)   % b�� ������ ���, x�� ���� �߻� ���ͷ� 

VL = var(x);                  % variance targeting, ǥ���л��� �����պл����� ��ü�Ͽ� ������ ��� ���� ���� 
V = zeros(length(x)-1,1);
V(1) = x(1)^2;                % ù ��° �л� �������� ù ��° ���ͷ��� �������� ���� 

for i = 2:length(x)-1
    % sigma(t)^2 = w + alpha*x(t-1)^2+beta*sigma(t-1)^2
    V(i) = (1-b(1)-b(2))*VL+b(1)*x(i)^2+b(2)*V(i-1);
end
%sigmat = V(length(x)-1)
%f = sum(log(V)+(x(2:end).^2)./V); % �쵵�Լ�
   
end