function price = ELS_pricing(S1min, S1max, S2min, S2max, sig1, sig2, rho, r, Dx, Dy, F, K0, c, K, KI, T, pp, Nx, Ny, method, knockin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving 2-D Black-Scholes PDE using FDM %
% - Using Alternating Directions Implicit (ADI) %
% - or Operator Splitting (OS) Method %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% S1min = min price of the underlying asset 1 in the space domain
% S1max = max price of the underlying asset 1 in the space domain
% S2min = min price of the underlying asset 2 in the space domain
% S2max = max price of the underlying asset 2 in the space domain
% sig1 = volatility of the underlying asset 1
% sig2 = volatility of the underlying asset 2
% rho = correlation between the prices of the two assets
% r = interest rate
% F = face value
% K0 = reference price of each asset
% c = rate of return on each early redemption date
% K = exercise price on each early redemption date
% KI = knock-in barrier level
% T = maturity of contract (year)
% pp = # of time points in 6 month
% Nx = # of space (X) points
% Ny = # of space (Y) points
% method = 0 for the ADI method, = 1 for the OS method
% knockin = 0 if a knock-in does not occur before maturity, = 1 otherwise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nt = 6*pp; % Total number of time points
dt = T/Nt; % Time step
h = (S1max - S1min)/Nx; % Space(X) step
k = (S2max - S2min)/Ny; % Space(Y) step

% Computational domain
x = linspace(S1min - 0.5*h, S1max + 0.5*h, Nx+2);
y = linspace(S2min - 0.5*h, S2max + 0.5*h, Ny+2);

% Pre-allocation
u(1:Nx+2, 1:Ny+2) = 0.0; u0 = u; d_x(1:Nx) = 0.0; d_y = d_x;

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
        else
            if knockin == 1 % If a knock-in occurs before maturity
                u0(i, j) = min(x(i)/K0(1), y(j)/K0(2))*F;
            elseif knockin == 0 % If a knock-in does not occur before maturity
                if min(x(i)/K0(1), y(j)/K0(2)) >= KI % If a knock-in does not occur at maturity
                    u0(i, j) = (1 + c(6))*F;
                else % If a knock-in does occur at maturity
                    u0(i, j) = min(x(i)/K0(1), y(j)/K0(2))*F;
                end
            end
        end
    end
end
u = u0;

% Linear boundary condition
u(1, 2:Ny+1) = 2*u(2, 2:Ny+1) - u(3, 2:Ny+1);
u(Nx+2, 2:Ny+1) = 2*u(Nx+1, 2:Ny+1) - u(Nx, 2:Ny+1);
u(1:Nx+2, 1) = 2*u(1:Nx+2, 2) - u(1:Nx+2, 3);
u(1:Nx+2, Ny+2) = 2*u(1:Nx+2, Ny+1) - u(1:Nx+2, Ny);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Time loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u2 = u;
for iter = 1:Nt
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% x - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 2:Ny+1
        for i = 2:Nx+1
            if method == 0 % ADI method
                d_y(i-1) = u(i,j)/dt + (1/4)*(sig2*y(j))^2*(u(i,j+1) - 2*u(i,j) + u(i,j-1))/(k^2) + (r-Dy)*y(j)*(u(i,j+1) - u(i,j))/(2*k) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(u(i+1,j+1) + u(i-1,j-1) - u(i-1,j+1) - u(i+1,j-1))/(4*h*k);
            elseif method == 1 % OS method
                d_y(i-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(u(i+1, j+1) - u(i+1, j-1) - u(i-1, j+1) + u(i-1, j-1))/(4*h*k) + u(i, j)/dt;
            end
        end
        u2(2:Nx+1, j) = tridiag(a_x,b_x, c_x, d_y);
    end
    
    % Linear boundary condition
    u2(1, 2:Ny+1) = 2*u2(2, 2:Ny+1) - u2(3, 2:Ny+1);
    u2(Nx+2, 2:Ny+1) = 2*u2(Nx+1, 2:Ny+1) - u2(Nx, 2:Ny+1);
    u2(1:Nx+2, 1) = 2*u2(1:Nx+2, 2) - u2(1:Nx+2, 3);
    u2(1:Nx+2, Ny+2) = 2*u2(1:Nx+2, Ny+1) - u2(1:Nx+2, Ny);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% y - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 2:Nx+1
        for j = 2:Ny+1
            if method == 0 % ADI method
               d_x(j-1) = u2(i,j)/dt + (sig1*x(i))^2*(u2(i+1,j) - 2*u2(i,j) + u2(i-1,j))/(4*h^2) + (r-Dx)*x(i)*(u2(i+1,j) - u2(i,j))/(2*h) + (1/2)*rho*sig1*sig2*x(i)*y(j)*(u2(i+1,j+1) + u2(i-1,j-1) - u2(i-1,j+1) - u2(i+1,j-1))/(4*h*k);
            elseif method == 1 % OS method
                d_x(j-1) = (1/2)*rho*sig1*sig2*x(i)*y(j)*(u2(i+1, j+1) - u2(i+1, j-1) - u2(i-1, j+1) + u2(i-1, j-1))/(4*h*k) + u2(i, j)/dt;
            end
        end
        u(i, 2:Ny+1) = tridiag(a_y, b_y, c_y, d_x);
    end
    
    % Linear boundary condition
    u(1, 2:Ny+1) = 2*u(2, 2:Ny+1) - u(3, 2:Ny+1);
    u(Nx+2, 2:Ny+1) = 2*u(Nx+1, 2:Ny+1) - u(Nx, 2:Ny+1);
    u(1:Nx+2, 1) = 2*u(1:Nx+2, 2) - u(1:Nx+2, 3);
    u(1:Nx+2, Ny+2) = 2*u(1:Nx+2, Ny+1) - u(1:Nx+2, Ny);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Early redemption %%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(iter,pp) == 0 && iter < Nt
        for i = 1:Nx+2
            for j = 1:Ny+2
                if min(x(i)/K0(1), y(j)/K0(2)) >= K(6 - floor(iter/pp))*F
                    u(i, j) = (1 + c(6 - floor(iter/pp)))*F;
                end
            end
        end
    end
end

price = u;

end