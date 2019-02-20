# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:59:18 2018

@author: rbgud

File : OSM을 이용한 ELS Pricing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Parameters setting
S1max = 7000; S1min = 0; # Max and min price of the underlying asset 1
S2max = 6000; S2min = 0; # Max and min price of the underlying asset 2
mu1 = 0.1; # mean return of the underlying asset 1
mu2 = 0.1; # mean return of the underlying asset 2
sig1 = 0.25; # Volatility of the underlying asset 1
sig2 = 0.2; # Volatility of the underlying asset 2
rho = 0.4; # Correlation between prices of the two assets
r = 0.02; # Interest rate
K0 = [3500, 3000]; # Reference price of each asset
F = 100; # Face value
T = 3; # Maturation of contract
c = [0.0285, 0.057, 0.0855, 0.114, 0.1425, 0.171]; # Rate of return on each early redemption date
K = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]; # Exercise price on each early redemption date
KI = 0.55; # Knock-In barrier level
Nt = 300;
Nx = 100;
Ny=Nx;
dt = T/Nt; 
hx = (S1max-S1min)/Nx; hy = hx; # (S2max-S2min)/Ny;

# Matrix initialize 
# x,y 가격 축 노드 생성 
x = np.linspace(S1min,S1max,Nx+2); y = np.linspace(S2min,S2max,Ny+2);

# matrix 초기화 
u0 = np.zeros((Nx+2,Ny+2))
u2 = np.zeros((Nx+2,Ny+2)) 

alpha_x= np.zeros(Nx);beta_x= np.zeros(Nx);gamma_x= np.zeros(Nx);
alpha_y= np.zeros(Ny);beta_y= np.zeros(Ny);gamma_y = np.zeros(Ny);
fx = np.zeros(Nx); 
fy = fx;

# Tridiagonal Matrix setting
# x의 tridiagonal matrix 계수 alpha, beta, gamma 설정
for i in range(1,Nx+1):
    beta_x[i-1] = 1/dt + (sig1*x[i])**2/(hx**2) + r*x[i]/hx + 0.5*r;
    alpha_x[i-1] = -0.5*(sig1*x[i])**2/(hx**2);
    gamma_x[i-1] = -0.5*(sig1*x[i])**2/(hx**2) - r*x[i]/hx;

#beta_x[0] = beta_x[0] + 2.0*alpha_x[0];     
#gamma_x[0] = gamma_x[0] - alpha_x[0]; 
#alpha_x[Nx-1] = alpha_x[Nx-1] - gamma_x[Nx-1]; 
#beta_x[Nx-1] = beta_x[Nx-1]+ 2.0*gamma_x[Nx-1];

# y의 tridiagonal matrix 계수 alpha, beta, gamma 설정
for j in range(1,Ny+1):
    beta_y[j-1] = 1/dt + (sig2*y[j])**2/(hy**2) + r*y[j]/hy + 0.5*r;
    alpha_y[j-1] = -0.5*(sig2*y[j])**2/(hy**2);
    gamma_y[j-1] = -0.5*(sig2*y[j])**2/(hy**2) - r*y[j]/hy;

#beta_y[0] = beta_y[0] + 2.0*alpha_y[0];     
#gamma_y[0] = gamma_y[0] - alpha_y[0]; 
#alpha_y[Ny-1] = alpha_y[Ny-1] - gamma_y[Ny-1]; 
#beta_y[Ny-1] = beta_y[Ny-1]+ 2.0*gamma_y[Ny-1];


# Initial condition 
for i in range(Nx+2):
    for j in range(Ny+2):
         if min(x[i]/K0[0],y[j]/K0[1]) <KI:
             u0[i,j] = F*min(x[i]/K0[0],y[j]/K0[1]);
         else:
             u0[i,j] = F*(1+c[5]);

u = u0;

       
#%% PDE Solving

             
def TDMAsolver(a, b, c, d):
 
    nf = len(a)     # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]

    xc = ac
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    del bc, cc, dc  # delete variables from memory

    return xc  

#%%

for k in range(Nt):
    # Early redemption condition 
    if k == 50:
        for i in range(Nx+2):
            for j in range(Ny+2):
                if (x[i]>= K0[0]*K[0] and y[j]>= K0[1]*K[0]):
                    u[i,j] = F*(1+c[0]);  
    if k == 100:                
        for i in range(Nx+2):
            for j in range(Ny+2):        
                if (x[i]>= K0[0]*K[1] and y[j]>= K0[1]*K[1]):
                    u[i,j] = F*(1+c[1]);
    if k == 150:                
        for i in range(Nx+2):
            for j in range(Ny+2):             
                if (x[i]>= K0[0]*K[2] and y[j]>= K0[1]*K[2]):
                   u[i,j] = F*(1+c[2]);
    if k == 200:               
        for i in range(Nx+2):
            for j in range(Ny+2):            
                if (x[i]>= K0[0]*K[3] and y[j]>= K0[1]*K[3]):
                    u[i,j] = F*(1+c[3]);
    if k == 250:                
        for i in range(Nx+2):
            for j in range(Ny+2):      
                if (x[i]>= K0[0]*K[4] and y[j]>= K0[1]*K[4]):
                   u[i,j] = F*(1+c[4]);
             
            
    # x - direction tridiagonal matrix solving
    for j in range(1,Ny+1):
        for i in range(1,Nx+1):
            fy[i-1] = 0.5*rho*sig1*sig2*x[i]*y[j]\
            *(u[i+1,j+1]-u[i+1,j]-u[i,j+1]+u[i,j])/(hx**2)+ u[i,j]/dt;
    
        u2[1:Nx+1,j]=TDMAsolver(alpha_x,beta_x,gamma_x,fy);
    
    # step1 Linear Boundary condition
    u2[0,1:Ny+1]=2*u2[1,1:Ny+1]-u2[2,1:Ny+1];
    u2[Nx+1,1:Ny+1]=2*u2[Nx,1:Ny+1]-u2[Nx-1,1:Ny+1];
    u2[0:Nx+1,0]=2*u2[0:Nx+1,1]-u2[0:Nx+1,2];
    u2[0:Nx+1,Ny+1]=2*u2[0:Nx+1,Ny]-u2[0:Nx+1,Ny-1];

    #%%
    # y - direction tridiagonal matrix solving
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
             fx[j-1] = 0.5*rho*sig1*sig2*x[i]*y[j]\
            *(u2[i+1,j+1]-u2[i+1,j]-u2[i,j+1]+u2[i,j])/(hy**2)+ u2[i,j]/dt;
       
        u[i,1:Ny+1]=TDMAsolver(alpha_y,beta_y,gamma_y,fx);
        
       
    # step2 Linear Boundary condition
    u[0,1:Ny+1]=2*u[1,1:Ny+1]-u[2,1:Ny+1];
    u[Nx+1,1:Ny+1]=2*u[Nx,1:Ny+1]-u[Nx-1,1:Ny+1];
    u[0:Nx+1,0]=2*u[0:Nx+1,1]-u[0:Nx+1,2];
    u[0:Nx+1,Ny+1]=2*u[0:Nx+1,Ny]-u[0:Nx+1,Ny-1];
  
#%%
#f1 = figure;
#[x,y] = np.meshgrid(x,y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#[x,y] = np.meshgrid(x,y)
ax.plot_mesh(x, y, u)
plt.show()
#f2 = figure;
#mesh(x, y, u);
fdm_price = u[round(Nx*K0[0]/S1max)+1,round(Ny*K0[1]/S2max)+1]

