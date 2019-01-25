%% GBM (Euler's Discretization) %%

S0 = 100;               % �ʱ��ְ�
mu = 0.1;               % �����ͷ�
vol= 0.3;               % ������(��)
T =1;                   % ��������
N = 365;                % �� time step ��
dt = T/N;
SP = zeros(1,N+1);      % �ְ� path �ʱ�ȭ
SP(1) = S0;             % ù° �� ���Ϳ� �ʱ��ְ� ����
w = randn(1,N);         % ���Ժ����� ������ ��������

%%

for i = 2:N+1
    SP(i) = SP(i-1)+SP(i-1)*(mu*dt+vol*w(i-1)*sqrt(dt));
end

plot(SP)
xlabel('time')
ylabel('Stock price')
