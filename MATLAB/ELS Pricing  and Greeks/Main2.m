clear all; clc; close all; format compact;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig1 = 0.2328; % Volatility of the underlying asset 1
sig2 = 0.2366; % Volatility of the underlying asset 2
rho = 0.5069; % Correlation between prices of the two assets
r = 0.0167; % Interest rate
K0 = [2056.50,3256.49]; % Reference price of each asset(EUROSTOXX50,S&P500)
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
Nt = 300;
Nx = 600;
Ny=600;
dt = T/Nt; 
hx = (S1max-S1min)/Nx; hy = (S2max-S2min)/Ny;
Nx0 = round(Nx*K0(1)/S1max)+1;
Ny0 = round(Ny*K0(2)/S2max)+1;
method = 1; % 0 for the ADI method, 1 for the OS method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Monte-Carlo simulation to compute the probatility that a knock-in occurs
% Monte-Carlo simulation is performed under P measure uisng mu1 and mu2

S1 = zeros(1,Nt); S2 = zeros(1,Nt);
S1(1) = K0(1); S2(1) = K0(2);

count = 0;
total_iter = 50000;
for iter = 1:total_iter
    for i = 1:Nt
        e1 = randn; e2 = randn;
%         S1(i+1) = S1(i)*(1 + r*dt + sig1*sqrt(dt)*e1);
%         S2(i+1) = S2(1)*(1 + r*dt + sig2*sqrt(dt)*(rho*e1 + sqrt(1-rho^2)*e2));
        S1(i+1)=S1(i)*exp((r-Dx-sig1^2/2)*dt+sig1*sqrt(dt)*e1);
        S2(i+1)=S2(i)*exp((r-Dy-sig2^2/2)*dt+sig2*sqrt(dt)*(rho*e1 + sqrt(1-rho^2)*e2));
    end
    
    if sum(min(S1/S1(1), S2/S2(1)) < KI) > 0
        count = count + 1;
    end
end

% probability that a knock-in occurs
prob = count/total_iter;

% ELS price if a knock-in does not occur
price_NKI = ELS_pricing(S1min, S1max, S2min, S2max, sig1, sig2, rho, r, Dx, Dy, F, K0, c, K, KI, T, pp, Nx, Ny, method, 0);

% ELS price if a knock-in occurs
price_KI = ELS_pricing(S1min, S1max, S2min, S2max, sig1, sig2, rho, r, Dx, Dy, F, K0, c, K, KI, T, pp, Nx, Ny, method, 1);

% Conditional expectation of ELS price
ELS_price = (1 - prob)*price_NKI + prob*price_KI;
ELS_price(Nx0,Ny0)  % ELS price at current underlying pricese observed
%%
% plotting 
x = linspace(S1min - 0.5*hx, S1max + 0.5*hx, Nx+2);
y = linspace(S2min - 0.5*hy, S2max + 0.5*hy, Ny+2);
mesh(x(2:Nx),y(2:Ny),ELS_price(2:Nx,2:Ny));    
hold on