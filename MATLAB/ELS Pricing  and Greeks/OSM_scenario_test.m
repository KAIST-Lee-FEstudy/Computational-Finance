%% Parameters setting
sig1 = 0.2328; % Volatility of the underlying asset 1
sig2 = 0.2366; % Volatility of the underlying asset 2
rho = 0.5069; % Correlation between prices of the two assets
r = 0.0167; % Interest rate
K0 = [2056.50,3256.49]; % Reference price of each asset(S&P500,EUROSTOXX50)
S1max = K0(1)*2; S1min = 0;  %Max and min price of the underlying asset 1
S2max = K0(2)*2; S2min = 0;  %Max and min price of the underlying asset 2
F = 100; % Face value
T = 3; % Maturation of contract
c = [0.033, 0.066, 0.099, 0.132, 0.165, 0.198]; % Rate of return on each early redemption date
K = [0.9 0.9 0.85 0.85 0.80 0.80]; % Exercise price on each early redemption date
KI = 0.5; % Knock-In barrier level
mu1 = 0.01; % mean return of the underlying asset 1
mu2 = 0.01; % mean return of the underlying asset 2
Dx = 0; % Dividend of first asset
Dy = 0; % Dividend of second asset
pp = 50;
Nt = 100;
Nx = 50;
Ny=50;
dt = T/Nt; 
h = (S1max - S1min)/Nx; % Space(X) step
k = (S2max - S2min)/Ny; % Space(Y) step
knockin = 0;
% Computational domain
x = linspace(S1min , S1max , Nx+2);
y = linspace(S2min , S2max , Ny+2);
sig1_scenario = linspace(0,1.5,Nx+2);
sig2_scenario = linspace(0,1.5,Ny+2);
rho_scenario = linspace(-1,1,Nx+2);
%% 변동성 변화에 따른 ELS 가격 변화 
ELS_Price1 = zeros(Nx+2,1);
ELS_Price2 = zeros(Ny+2,1);
ELS_Price3 = zeros(Nx+2,Ny+2);
ELS_Price4 = zeros(Nx+2,Ny+2);
for i=1:length(sig1_scenario)
    ELS_Price1(i) = OSM_ELS_Price(S1max,S1min,S2max,S2min,sig1_scenario(i),sig2,rho,r,K0,F,T,c,K,KI);
    ELS_Price2(i) = OSM_ELS_Price(S1max,S1min,S2max,S2min,sig1,sig2_scenario(i),rho,r,K0,F,T,c,K,KI);
end
for i = 1: length(sig1_scenario)
    for j = 1 : length(sig2_scenario)
        ELS_Price3(i,j) = OSM_ELS_Price(S1max,S1min,S2max,S2min,sig1_scenario(i),sig2_scenario(j)...
                          ,rho,r,K0,F,T,c,K,KI);
    end
end
for i = 1: length(sig1_scenario)
    for j = 1 : length(rho_scenario)
        ELS_Price4(i,j) = OSM_ELS_Price(S1max,S1min,S2max,S2min,sig1_scenario(i),sig2...
                          ,rho_scenario(j),r,K0,F,T,c,K,KI);
    end
end
        

%% Backward scheme을 이용한 Vega 계산 
Vega_x(1,Nx+2) = 0.0;
Vega_y(1,Nx+2) = 0.0;

for i = 2: Nx+1
    Vega_x(i-1) = (ELS_Price1(i-1)-ELS_Price1(i));%/(sig1_scenario(i)-sig1_scenario(i-1));
    Vega_y(i-1) = (ELS_Price2(i-1)-ELS_Price2(i));%/(sig2_scenario(i)-sig2_scenario(i-1));
end
Vega_x(Vega_x==0)=nan;
Vega_y(Vega_y==0)=nan;
%% 상관계수 변화에 따른 ELS 가격 변화
ELS_Price_corr_chg = zeros(Nx+2,1);

for i=1:length(rho_scenario)
    ELS_Price_corr_chg(i) = OSM_ELS_Price(S1min, S1max, S2min, S2max, sig1, sig2, rho_scenario(i), r, Dx, Dy, F, K0, c, K, KI, T, pp);
end

%%
f1 = figure;
plot(sig1_scenario,ELS_Price1)
title('EUROSTOXX50 변동성 변화에 따른 ELS 가격 변화')
xlabel('EUROSTOXX50 변동성')
ylabel('ELS 가격')

f2 = figure;
mesh(sig1_scenario,sig2_scenario,ELS_Price3)
title('EUROSTOXX50, S&P500 변동성 시나리오에 따른 ELS 가격 변화')
xlabel('EUROSTOXX50 변동성')
ylabel('S&P500 변동성')
zlabel('ELS 가격')
f2 = figure;
mesh(sig1_scenario,rho_scenario,ELS_Price4)
title('EUROSTOXX50 변동성, 상관계수 시나리오에 따른 ELS 가격 변화\n(S&P500 변동성 고정)')
xlabel('EUROSTOXX50 변동성')
ylabel('상관계수')
zlabel('ELS 가격')

f3 = figure;
plot(sig2_scenario,ELS_Price2)
title('S&P500 변동성 변화에 따른 ELS 가격 변화')
xlabel('S&P500 변동성')
ylabel('ELS 가격')

x = linspace(S1min,S1max,Nx+2);
y = linspace(S2min,S2max,Ny+2);
f2 =figure;
plot(x,Vega_x*100)
hold on
line([K0(2)*KI K0(2)*KI],get(gca,'YLim'),'Color','r');
legend('ELS Vega with respect to EUROSTOXX50')
xlabel('EUROSTOXX50')
ylabel('ELS Vega')
title('Vega of ELS ')

f3 =figure;
plot(y,Vega_y*100)
hold on
line([K0(2)*KI K0(2)*KI],get(gca,'YLim'),'Color','r');
legend('ELS Vega with respect to S&P500')
xlabel('S&P500')
ylabel('ELS Vega')
title('Vega of ELS ')
%%
x = linspace(S1min,S1max,Nx+2);
y = linspace(S2min,S2max,Ny+2);
Corr_chg = zeros(Nx+2,1);
for i = 2: Nx+1
    Corr_chg(i-1) = (ELS_Price_corr_chg(i-1)-ELS_Price_corr_chg(i));
end
Corr_chg(Corr_chg==0)=nan;
f4 = figure;
plot(rho_scenario,ELS_Price_corr_chg)
title('상관계수에 대한 ELS 가격 민감도(변동성 고정)')
xlabel('Correlation')
ylabel('ELS Price')
% 
f5 = figure;
plot(x,Corr_chg/(rho_scenario(2)-rho_scenario(1)))
title('상관계수에 대한 ELS 가격 민감도(변동성 고정)')
xlabel('Underlying Asset Price')
ylabel('Rega(Correlation sensitivity)')


    