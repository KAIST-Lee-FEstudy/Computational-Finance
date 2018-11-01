%%%%%%%% Local Volatility Surface %%%%%%%%
clear all; clc;
% 초기 입력변수
C_P ='C';
S = 292.42;     % 기초자산 현재가격
r = 0.0165;
q = 0.017;

%옵션 거래데이터
market_data = xlsread('ksp200call2.xlsx');
temp_size = size(market_data);
X = market_data(:,1);               % 행사가격
T = market_data(:,2);               % 잔존만기
P = market_data(:,3);               % 시장가격

% 1) 행사가격/잔존만기별 내재변동성 추정(뉴턴랩슨 or 바이섹션)
Imp_Vol = zeros(temp_size(1),1);
for i = 1:temp_size(1)
    Imp_Vol(i) =  NewtonRaphson_ImpVol(C_P,S,X(i),r,T(i),q,P(i));
end
Imp_Vol(~isfinite(Imp_Vol))= NaN;
Imp_Vol = fillmissing(Imp_Vol,'nearest');
%%
% 2) 내재변동성 함수 모델설정 및 계수추정 
M = log((S*exp((r-q)*T))./X);                 % Moneyness
Mdata = [M,T];
% Least squre method
a = lsqcurvefit(@ImpVol_f,[0.1 0.1 0.1 0.1 0.1 0.1 0.1],Mdata,Imp_Vol);
%%
% Moneyness , 잔존만기의 범위 설정 (구할 내재변동성의 범위를 의미)
M = linspace(min(M),max(M),365);
% T = linspace(min(T),max(T),365);
T = linspace(0,3,365);
T2 = linspace(min(T),max(T),365);
%%
% 해당 Moneyness, 잔존만기의 범위에 따른 행사가격
X = (S*exp((r-q)*T))./exp(M);

% 3) 내재변동성 함수이용 도함수 계산

Imp_Vol = zeros(length(M),length(T));
dT = zeros(length(M),length(T));
dX = zeros(length(M),length(T));
dXX = zeros(length(M),length(T));
d = zeros(length(M),length(T));
for i = 1: length(M)
    for j = 1: length(T)
        Imp_Vol(i,j) = ImpVol_f(a,[M(i) T(j)]);
        dT(i,j) = a(3)*a(2)*exp(a(2)*T(j))*(r-q)*(a(4)+2*a(5)*M(i)+3*a(6)*M(i)^2....
        +4*a(7)*M(i)^3);
        dX(i,j) = -(a(4)+2*a(5)*M(i)+3*a(6)*M(i)^2+4*a(7)*M(i)^3)/X(i);
        dXX(i,j) = (a(4)+2*a(5)*(M(i)+1)+3*a(6)*M(i)*(M(i)+2)+...
                   4*a(7)*M(i)^2*(M(i)+3))/X(i)^2;
        d(i,j) = (log(S/X(i))+(r-q+0.5*Imp_Vol(i,j)^2)* ...
                 T(j))/(Imp_Vol(i,j)*sqrt(T(j)));
    end
end

% 4) Local Volatility Surface 계산 
Local_Vol = zeros(length(X),length(T));
for i = 1:length(X)
    for j =1:length(T)
        Local_Vol(i,j) = (Imp_Vol(i,j)^2+2*Imp_Vol(i,j)*T(j)*...
               (dT(i,j)+(r-q)*X(i)*dX(i,j)))/((1+X(i)*d(i,j)*dX(i,j)*...
               sqrt(T(j)))^2+Imp_Vol(i,j)^2*X(i)^2*T(j)*(dXX(i,j)-d(i,j)*...
               dX(i,j)^2*sqrt(T(j))));
    end
end
Local_Vol = sqrt(Local_Vol);

%%
% Plot
[X,T] = meshgrid(X,T);
% [M,T2] = meshgrid(M,T2);
Imp_Vol = Imp_Vol';
Local_Vol = Local_Vol';

figure (1)                  % Implied Volatility Surface
mesh(X,T,Imp_Vol)
title('Implied Volatility')
xlabel('Strike price')
ylabel('Time to Maturity')
zlabel('Volatility')
figure (2)                  % Local Volatility Surface
mesh(X,T,Local_Vol)
title('Local Volatility')
xlabel('Strike price')
ylabel('Time to Maturity')
zlabel('Volatility')
figure (3)                  
mesh(M,T2,Imp_Vol)
title('Implied Volatility')
ylabel('Time to Maturity')
xlabel('Moneyness')
zlabel('Volatility')
figure (4)                  
mesh(M,T2,Local_Vol)
title('Local Volatility')
ylabel('Time to Maturity')
xlabel('Moneyness')
zlabel('Volatility')           
           
          
        
