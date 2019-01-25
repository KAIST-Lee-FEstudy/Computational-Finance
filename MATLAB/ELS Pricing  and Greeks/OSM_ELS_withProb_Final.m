clear all;clc;close all;
%% Parameters setting
sig1 = 0.2328; % Volatility of the underlying asset 1
sig2 = 0.2366; % Volatility of the underlying asset 2
rho = 0.5069; % Correlation between prices of the two assets
r = 0.0167; % Interest rate
K0 = [2056.50,3256.49]; % Reference price of each asset(S&P500,EUROSTOXX50)
S1max = K0(1)*2; S1min = 0;  %Max and min price of the underlying asset 1
S2max = K0(2)*2; S2min = 0;  %Max and min price of the underlying asset 2
F = 10000; % Face value
T = 3; % Maturation of contract
c = [0.033, 0.066, 0.099, 0.132, 0.165, 0.198]; % Rate of return on each early redemption date
K = [0.9 0.9 0.85 0.85 0.80 0.80]; % Exercise price on each early redemption date
KI = 0.5; % Knock-In barrier level
mu1 = 0.01; % mean return of the underlying asset 1
mu2 = 0.01; % mean return of the underlying asset 2
Dx = 0; % Dividend of first asset
Dy = 0; % Dividend of second asset
pp = 100;
Nt = 600;
Nx = 100;
Ny=100;
dt = T/Nt; 
h = (S1max - S1min)/Nx; % Space(X) step
k = (S2max - S2min)/Ny; % Space(Y) step
knockin = 0;
method =1;

% Computational domain
x = linspace(S1min , S1max , Nx+2);
y = linspace(S2min , S2max , Ny+2);
% Pre-allocation
u(1:Nx+2, 1:Ny+2) = 0.0; u0 = u; d_x(1:Nx) = 0.0; d_y = d_x;d_yy =d_x; d_xx = d_x;
uu(1:Nx+2, 1:Ny+2) = 0.0; u00 = u; d_x(1:Nx) = 0.0;

% Coefficients of x (for the first step of the ADI/OS algorithm)
a_x = zeros(1,Nx); b_x = zeros(1,Nx); c_x = zeros(1,Nx);
for i = 2:Nx+1
    if method == 0
        b_x(i-1) = -(sig1*x(i))^2/(4*h^2);
        c_x(i-1) = -(sig1*x(i))^2/(4*h^2) - (r - Dx)*x(i)/(2*h);
        a_x(i-1) = 1/dt + (sig1*x(i))^2/(2*h^2) + (r - Dx)*x(i)/(2*h) + r/2; 
    elseif method == 1
    b_x(i-1) = -(sig1*x(i))^2/(2*h^2);
    c_x(i-1) = -(sig1*x(i))^2/(2*h^2) - (r - Dx)*x(i)/h;
    a_x(i-1) = 1/dt + (sig1*x(i))^2/(h^2) + (r - Dx)*x(i)/h + r/2;
    end
end
a_x(1) = a_x(1) + 2.0*b_x(1);
c_x(1) = c_x(1) - b_x(1);
b_x(Nx) = b_x(Nx) - c_x(Nx);
a_x(Nx) = a_x(Nx) + 2.0*c_x(Nx);

% Coefficient y (for the second step of the ADI/OS algorithm)
a_y = zeros(1,Ny); b_y = zeros(1,Ny); c_y = zeros(1,Ny);
for j = 2:Ny+1
    if method == 0 % ADI method
        b_y(j-1) = -(sig2*y(j))^2/(4*k^2);
        c_y(j-1) = -(sig2*y(j))^2/(4*k^2) - (r - Dy)*y(j)/(2*k);
        a_y(j-1) = 1/dt + (sig2*y(j))^2/(2*k^2) + (r - Dy)*y(j)/(2*k) + r/2;
    elseif method == 1 % OS method
    b_y(j-1) = -(sig2*y(j))^2/(2*k^2);
    c_y(j-1) = -(sig2*y(j))^2/(2*k^2) - r*y(j)/k;
    a_y(j-1) = 1/dt + (sig2*y(j))^2/(k^2) + (r - Dy)*y(j)/k + r/2;
    end
end
a_y(1) = a_y(1) + 2.0*b_y(1);
c_y(1) = c_y(1) - b_y(1);
b_y(Ny) = b_y(Ny) - c_y(Ny);
a_y(Ny) = a_y(Ny) + 2.0*c_y(Ny);

% Initial condition
for i = 1:Nx+2
    for j = 1:Ny+2
        if min(x(i)/K0(1), y(j)/K0(2)) >= K(6)
            u0(i, j) = (1 + c(6))*F;
            u00(i, j) = (1 + c(6))*F;
        else
            u00(i, j) = min(x(i)/K0(1), y(j)/K0(2))*F;
            if knockin == 0 % If a knock-in does not occur before maturity
                if min(x(i)/K0(1), y(j)/K0(2)) >= KI % If a knock-in does not occur at maturity
                    u0(i, j) = (1 + c(6))*F;
                else % If a knock-in does occur at maturity
                    u0(i, j) = min(x(i)/K0(1), y(j)/K0(2))*F;
                end
            end
        end
    end
end
u = u0; uu = u00;

% Linear boundary condition
u(1, 2:Ny+1) = 2*u(2, 2:Ny+1) - u(3, 2:Ny+1);
u(Nx+2, 2:Ny+1) = 2*u(Nx+1, 2:Ny+1) - u(Nx, 2:Ny+1);
u(1:Nx+2, 1) = 2*u(1:Nx+2, 2) - u(1:Nx+2, 3);
u(1:Nx+2, Ny+2) = 2*u(1:Nx+2, Ny+1) - u(1:Nx+2, Ny);

uu(1, 2:Ny+1) = 2*uu(2, 2:Ny+1) - uu(3, 2:Ny+1);
uu(Nx+2, 2:Ny+1) = 2*uu(Nx+1, 2:Ny+1) - uu(Nx, 2:Ny+1);
uu(1:Nx+2, 1) = 2*uu(1:Nx+2, 2) - u(1:Nx+2, 3);
uu(1:Nx+2, Ny+2) = 2*uu(1:Nx+2, Ny+1) - uu(1:Nx+2, Ny);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Time loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u2 = u;u22 = uu;
for iter = 1:Nt
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% x - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 2:Ny+1
        for i = 2:Nx+1
            if method == 0 % ADI method
                d_y(i-1) = u(i,j)/dt + (1/4)*(sig2*y(j))^2*(u(i,j+1) - 2*u(i,j) + u(i,j-1))/(k^2) + (r-Dy)*y(j)*(u(i,j+1) - u(i,j))/(2*k) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(u(i+1,j+1) + u(i-1,j-1) - u(i-1,j+1) - u(i+1,j-1))/(4*h*k);
                d_yy(i-1) = uu(i,j)/dt + (1/4)*(sig2*y(j))^2*(uu(i,j+1) - 2*uu(i,j) + uu(i,j-1))/(k^2) + (r-Dy)*y(j)*(uu(i,j+1) - uu(i,j))/(2*k) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(uu(i+1,j+1) + uu(i-1,j-1) - uu(i-1,j+1) - uu(i+1,j-1))/(4*h*k);
            elseif method == 1 % OS method
            end
            d_y(i-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(u(i+1, j+1) - u(i+1, j-1) - u(i-1, j+1) + u(i-1, j-1))/(4*h*k) + u(i, j)/dt;
            d_yy(i-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(uu(i+1, j+1) - uu(i+1, j-1) - uu(i-1, j+1) + uu(i-1, j-1))/(4*h*k) + uu(i, j)/dt;
        end
        u2(2:Nx+1, j) = tridiag(a_x,b_x, c_x, d_y);
        u22(2:Nx+1, j) = tridiag(a_x,b_x, c_x, d_yy);
    end
    
    % Linear boundary condition
    u2(1, 2:Ny+1) = 2*u2(2, 2:Ny+1) - u2(3, 2:Ny+1);
    u2(Nx+2, 2:Ny+1) = 2*u2(Nx+1, 2:Ny+1) - u2(Nx, 2:Ny+1);
    u2(1:Nx+2, 1) = 2*u2(1:Nx+2, 2) - u2(1:Nx+2, 3);
    u2(1:Nx+2, Ny+2) = 2*u2(1:Nx+2, Ny+1) - u2(1:Nx+2, Ny);
    
    u22(1, 2:Ny+1) = 2*u22(2, 2:Ny+1) - u22(3, 2:Ny+1);
    u22(Nx+2, 2:Ny+1) = 2*u22(Nx+1, 2:Ny+1) - u22(Nx, 2:Ny+1);
    u22(1:Nx+2, 1) = 2*u22(1:Nx+2, 2) - u22(1:Nx+2, 3);
    u22(1:Nx+2, Ny+2) = 2*u22(1:Nx+2, Ny+1) - u22(1:Nx+2, Ny);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% y - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 2:Nx+1
        for j = 2:Ny+1
            if method == 0 % ADI method
               d_x(j-1) = u2(i,j)/dt + (sig1*x(i))^2*(u2(i+1,j) - 2*u2(i,j) + u2(i-1,j))/(4*h^2) + (r-Dx)*x(i)*(u2(i+1,j) - u2(i,j))/(2*h) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(u2(i+1,j+1) + u2(i-1,j-1) - u2(i-1,j+1) - u2(i+1,j-1))/(4*h*k);
               d_xx(j-1) = u22(i,j)/dt + (sig1*x(i))^2*(u22(i+1,j) - 2*u22(i,j) + u22(i-1,j))/(4*h^2) + (r-Dx)*x(i)*(u22(i+1,j) - u22(i,j))/(2*h) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(u22(i+1,j+1) + u22(i-1,j-1) - u22(i-1,j+1) - u22(i+1,j-1))/(4*h*k);
            elseif method == 1 % OS method
            end
            d_x(j-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(u2(i+1, j+1) - u2(i+1, j-1) - u2(i-1, j+1) + u2(i-1, j-1))/(4*h*k) + u2(i, j)/dt;
            d_xx(j-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(u22(i+1, j+1) - u22(i+1, j-1) - u22(i-1, j+1) + u22(i-1, j-1))/(4*h*k) + u22(i, j)/dt;
        end
        u(i, 2:Ny+1) = tridiag(a_y, b_y, c_y, d_x);
        uu(i, 2:Ny+1) = tridiag(a_y, b_y, c_y, d_xx);
    end
    
    % Linear boundary condition
    u(1, 2:Ny+1) = 2*u(2, 2:Ny+1) - u(3, 2:Ny+1);
    u(Nx+2, 2:Ny+1) = 2*u(Nx+1, 2:Ny+1) - u(Nx, 2:Ny+1);
    u(1:Nx+2, 1) = 2*u(1:Nx+2, 2) - u(1:Nx+2, 3);
    u(1:Nx+2, Ny+2) = 2*u(1:Nx+2, Ny+1) - u(1:Nx+2, Ny);
    
    uu(1, 2:Ny+1) = 2*uu(2, 2:Ny+1) - uu(3, 2:Ny+1);
    uu(Nx+2, 2:Ny+1) = 2*uu(Nx+1, 2:Ny+1) - uu(Nx, 2:Ny+1);
    uu(1:Nx+2, 1) = 2*uu(1:Nx+2, 2) - uu(1:Nx+2, 3);
    uu(1:Nx+2, Ny+2) = 2*uu(1:Nx+2, Ny+1) - uu(1:Nx+2, Ny);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Early redemption %%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(iter,pp) == 0 && iter < Nt
        for i = 1:Nx+2
            for j = 1:Ny+2
                if min(x(i)/K0(1), y(j)/K0(2)) >= K(6 - floor(iter/pp))*F
                    u(i, j) = (1 + c(6 - floor(iter/pp)))*F;
                    uu(i, j) = (1 + c(6 - floor(iter/pp)))*F;
                end
            end
        end
    end
end
% Monte-Carlo simulation to compute the probatility that a knock-in occurs
% Monte-Carlo simulation is performed under P measure uisng mu1 and mu2

S1 = zeros(1,Nt); S2 = zeros(1,Nt);
S1(1) = K0(1); S2(1) = K0(2);

count = 0;
total_iter = 50000;
for iter = 1:total_iter
    for i = 1:Nt
        e1 = randn; e2 = randn;
        S1(i+1)=S1(i)*exp((r-Dx-sig1^2/2)*dt+sig1*sqrt(dt)*e1);
        S2(i+1)=S2(i)*exp((r-Dy-sig2^2/2)*dt+sig2*sqrt(dt)*(rho*e1 + sqrt(1-rho^2)*e2));
    end
    
    if sum(min(S1/S1(1), S2/S2(1)) < KI) > 0
        count = count + 1;
    end
end

% probability that a knock-in occurs
prob = count/total_iter;
Total_ELSPrice = (1-prob)*u + prob*uu;
fdm_price = Total_ELSPrice(round(Nx*K0(1)/S1max)+1,round(Ny*K0(2)/S2max)+1);
%%
fprintf('OSM ELS Price : %.4f\n', fdm_price)
f1 = figure;
mesh(x, y, u0);
title('Initial condition KI No Hit')
xlabel('EUROSTOXX50')
ylabel('S&P500')
zlabel('ELS Payoff')
f2 = figure;
mesh(x, y, u00);
title('Initial condition KI Hit')
xlabel('EUROSTOXX50')
ylabel('S&P500')
zlabel('ELS Payoff')
f3 = figure;
mesh(x, y, Total_ELSPrice);
title('ELS Price')
xlabel('EUROSTOXX50')
ylabel('S&P500')
zlabel('ELS Price')
%% Greek 계산 
% Delta는 Backward difference scheme을 이용하여 계산
% Gamma는 Central difference scheme을 이용하여 계산 
Delta_x(1,Nx+2) = 0.0;
Delta_y(1,Nx+2) = 0.0;
Gamma_x(1,Nx+2) = 0.0;
Gamma_y(1,Nx+2) = 0.0;

for i = 2: Nx+1
    Delta_x(i-1) = (Total_ELSPrice(i,round(Ny*K0(2)/S2max)+1)-Total_ELSPrice(i-1,round(Ny*K0(2)/S2max)+1))/h;
    Delta_y(i-1) = (Total_ELSPrice(round(Nx*K0(1)/S1max)+1,i)-Total_ELSPrice(round(Nx*K0(1)/S1max)+1,i-1))/k;
    Gamma_x(i-1) = (u(i+1,round(Ny*K0(2)/S2max)+1)-2*u(i,round(Ny*K0(2)/S2max)+1)+u(i-1,round(Ny*K0(2)/S2max)+1))/(h^2);
    Gamma_y(i-1) = (u(round(Nx*K0(1)/S1max)+1,i+1)-2*u(round(Nx*K0(1)/S1max)+1,i)+u(i-1,round(Nx*K0(1)/S1max)+1))/(k^2);
end
Gamma_x(Gamma_x==0)=nan;
Gamma_y(Gamma_y==0)=nan;
%%
f4 = figure;
Delta_x = meshgrid(Delta_x);
mesh(x,y,Delta_x)
xlabel('S&P500')
ylabel('EUROSTOXX50')
zlabel('ELS Delta')
title('ELS Delta with respect to S&P500')
% plot(x,Delta_x)
% hold on
% line([K0(1)*KI K0(1)*KI],get(gca,'YLim'),'Color','r');
% xlabel('S&P500')
% ylabel('ELS Delta')
% title('ELS Delta with respect to S&P500')
f5 = figure;
Delta_y= meshgrid(Delta_y);
mesh(x,y,Delta_y)
ylabel('S&P500')
xlabel('EUROSTOXX50')
zlabel('ELS Delta')
title('ELS Delta with respect to EUROSTOXX50')
% plot(y,Delta_y)
% hold on
% line([K0(2)*KI K0(2)*KI],get(gca,'YLim'),'Color','r');
% xlabel('EUROSTOXX50')
% ylabel('ELS Delta')
% title('ELS Delta with respect to EUROSTOXX50')
f6 = figure;
Gamma_x = meshgrid(Gamma_x);
mesh(x,y,Gamma_x)
xlabel('S&P500')
ylabel('EUROSTOXX50')
zlabel('ELS Gamma')
title('ELS Gamma with respect to S&P500')
% plot(x,Gamma_x)
% hold on
% line([K0(1)*KI K0(1)*KI],get(gca,'YLim'),'Color','r');
% xlabel('S&P500')
% ylabel('ELS Gamma')
% title('ELS Gamma with respect to S&P500')
f7 = figure;
Gamma_y = meshgrid(Gamma_y);
mesh(x,y,Gamma_y)
ylabel('S&P500')
xlabel('EUROSTOXX50')
zlabel('ELS Gamma')
title('ELS Gamma with respect to EUROSTOXX50')
% plot(y,Gamma_y)
% hold on
% line([K0(2)*KI K0(2)*KI],get(gca,'YLim'),'Color','r');
% xlabel('EUROSTOXX50')
% ylabel('ELS Gamma')
% title('ELS Gamma with respect to EUROSTOXX50')
%%
f8 = figure;
mesh(x,y,uu)
title('ELS Price KI Hit')
xlabel('S&P500')
ylabel('EUROSTOXX50')
zlabel('ELS Price')
%%
f9 =figure;
mesh(x,y,u)
title('ELS Price KI No Hit')
xlabel('S&P500')
ylabel('EUROSTOXX50')
zlabel('ELS Price')
