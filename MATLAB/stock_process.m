%% GBM (Euler's Discretization) %%

S0 = 100;               % 초기주가
mu = 0.1;               % 기대수익률
vol= 0.3;               % 변동성(연)
T =1;                   % 잔존만기
N = 365;                % 총 time step 수
dt = T/N;
SP = zeros(1,N+1);      % 주가 path 초기화
SP(1) = S0;             % 첫째 열 벡터에 초기주가 세팅
w = randn(1,N);         % 정규분포를 따르는 난수생성

%%

for i = 2:N+1
    SP(i) = SP(i-1)+SP(i-1)*(mu*dt+vol*w(i-1)*sqrt(dt));
end

plot(SP)
xlabel('time')
ylabel('Stock price')
