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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#%% Parameters setting
sig1 = 0.2328 # Volatility of the underlying asset 1
sig2 = 0.2366 # Volatility of the underlying asset 2
rho = 0.5069 # Correlation between prices of the two assets
r = 0.0167 # Interest rate
K0 = [2056.50,3256.49] # Reference price of each asset(S&P500,EUROSTOXX50)
S1max = K0[0]*2; S1min = 0  # Max and min price of the underlying asset 1
S2max = K0[1]*2; S2min = 0  # Max and min price of the underlying asset 2
F = 10000 # Face value
T = 3 # Maturation of contract
c = [0.033, 0.066, 0.099, 0.132, 0.165, 0.198] # Rate of return on each early redemption date
K = [0.9, 0.9, 0.85, 0.85, 0.80, 0.80] # Exercise price on each early redemption date
KI = 0.5 # Knock-In barrier level
mu1 = 0.01 # mean return of the underlying asset 1
mu2 = 0.01 # mean return of the underlying asset 2
Dx = 0 # Dividend of first asset
Dy = 0 # Dividend of second asset
pp = 100
Nt = 600
Nx = 100
Ny=100
dt = T/Nt 
h = (S1max - S1min)/Nx # Space(X) step
k = (S2max - S2min)/Ny # Space(Y) step
knockin = 0
method =1

#%%
# Computational domain
x = np.linspace(S1min , S1max , Nx+2)
y = np.linspace(S2min , S2max , Ny+2)
#Pre-allocation
u = np.zeros((Nx+2,Ny+2)); u0 = u; d_x = np.zeros(Nx); d_y = d_x; d_yy =d_x; d_xx = d_x;
uu = np.zeros((Nx+2,Ny+2)); u00 = u; 

#%%
# Coefficients of x (for the first step of the ADI/OS algorithm)
a_x = np.zeros(Nx); b_x = np.zeros(Nx); c_x = np.zeros(Nx);

for i in range(1,Nx+1):
    if method == 0:
        b_x[i-1] = -(sig1*x[i])**2/(4*(h**2))
        c_x[i-1] = -(sig1*x[i])**2/(4*(h**2)) - (r - Dx)*x[i]/(2*h)
        a_x[i-1] = 1/dt + (sig1*x[i])**2/(2*(h**2)) + (r - Dx)*x[i]/(2*h) + r/2
    elif method == 1:
        b_x[i-1] = -(sig1*x[i])**2/(2*(h**2))
        c_x[i-1] = -(sig1*x[i])**2/(2*(h**2)) - (r - Dx)*x[i]/h
        a_x[i-1] = 1/dt + (sig1*x[i])**2/(h**2) + (r - Dx)*x[i]/h + r/2
    
a_x[0] = a_x[0] + 2.0*b_x[0]
c_x[0] = c_x[0] - b_x[0]
b_x[Nx-1] = b_x[Nx-1] - c_x[Nx-1]
a_x[Nx-1] = a_x[Nx-1] + 2.0*c_x[Nx-1]

# Coefficient y (for the second step of the ADI/OS algorithm)
a_y = np.zeros(Ny); b_y = np.zeros(Ny); c_y = np.zeros(Ny);

for j in range(1,Ny+1):
    if method == 0:
        b_y[j-1] = -(sig2*y[j])**2/(4*(k**2)) #alpha
        c_y[j-1] = -(sig2*y[j])**2/(4*(k**2)) - (r - Dy)*y[j]/(2*k) #gamma
        a_y[j-1] = 1/dt + (sig2*y[j])**2/(2*(k**2)) + (r - Dy)*y[j]/(2*k) + r/2 #beta
    elif method == 1:
        b_y[j-1] = -(sig2*y[j])**2/(2*(k**2))
        c_y[j-1] = -(sig2*y[j])**2/(2*(k**2)) - (r - Dy)*y[j]/k
        a_y[j-1] = 1/dt + (sig2*y[j])**2/(k**2) + (r - Dy)*y[j]/k + r/2
    
a_y[0] = a_y[0] + 2.0*b_y[0]
c_y[0] = c_y[0] - b_y[0]
b_y[Nx-1] = b_y[Nx-1] - c_y[Nx-1]
a_y[Nx-1] = a_y[Nx-1] + 2.0*c_y[Nx-1]

# Initial condition
for i in range(Nx+2):
    for j in range(Ny+2):
        if min(x[i]/K0[0], y[j]/K0[1]) >= K[5]:
            u0[i, j] = (1 + c[5])*F
            u00[i, j] = (1 + c[5])*F
        else :
            u00[i, j] = min(x[i]/K0[0], y[j]/K0[1])*F
            if knockin == 0: # If a knock-in does not occur before maturity
                if min(x[i]/K0[0], y[j]/K0[1]) >= KI: # If a knock-in does not occur at maturity
                    u0[i, j] = (1 + c[5])*F
                else: # If a knock-in does occur at maturity
                    u0[i, j] = min(x[i]/K0[0], y[j]/K0[1])*F

u = u0; uu = u00;

# Linear boundary condition
u[0,1:Ny+1] = 2*u[1,1:Ny+1] - u[2,1:Ny+1]
u[Nx+1, 1:Ny+1] = 2*u[Nx, 1:Ny+1] - u[Nx-1, 1:Ny+1]
u[0:Nx+2,0] = 2*u[0:Nx+2,1] - u[0:Nx+2,2]
u[0:Nx+2, Ny+1] = 2*u[0:Nx+2, Ny] - u[0:Nx+2, Ny-1]

uu[0, 1:Ny+1] = 2*uu[1, 1:Ny+1] - uu[2, 1:Ny+1]
uu[Nx+1, 1:Ny+1] = 2*uu[Nx, 1:Ny+1] - uu[Nx-1, 1:Ny+1]
uu[0:Nx+2,0] = 2*uu[0:Nx+2,1] - uu[0:Nx+2,2]
uu[0:Nx+2, Ny+1] = 2*uu[0:Nx+2, Ny] - uu[0:Nx+2,Ny-1]
# Time loop
u2 = u;u22 = uu;

#%%
def TDMAsolver(a, b, c, d):
 
        nf = len(d)     # number of edivuations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]
    
        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    
#        del bc, cc, dc  # delete variables from memory
    
        return xc  
    
#%%
iteration=0
#while iteration < Nt :
for time in range(Nt):
    # x - direction 
    for j in range (1,Ny+1):
        
        for i in range(1,Ny+1):
            if method == 0: # ADI method
                d_y[i-1] = u[i,j]/dt + (1/4)*((sig2*y[j])**2)*(u[i,j+1] - 2*u[i,j] + u[i,j-1])/(k**2) \
                            + (r-Dy)*y[j]*(u[i,j+1]- u[i,j])/(2**k) \
                            + (1/2)*rho*sig1*sig2*x[i]*y[j]*(u[i+1,j+1] + u[i-1,j-1] - u[i-1,j+1] - u[i+1,j-1])/(4*h*k)
                
                d_yy[i-1] = uu[i,j]/dt + (1/4)*((sig2*y[j])**2)*(uu[i,j+1] - 2*uu[i,j] + uu[i,j-1])/(k**2) \
                            + (r-Dy)*y[j]*(uu[i,j+1]- uu[i,j])/(2*k)\
                            + (1/2)*rho*sig1*sig2*x[i]*y[j]*(uu[i+1,j+1] + uu[i-1,j-1] - uu[i-1,j+1] - uu[i+1,j-1])/(4*h*k)
            elif method == 1: # OS method
            
                 d_y[i-1] = (1/2)*rho*sig1*sig2*x[i]*y[j]*(u[i+1, j+1]\
                            - u[i+1, j-1] - u[i-1, j+1] + u[i-1, j-1])/(4*h*k) + u[i, j]/dt
                 d_yy[i-1] = (1/2)*rho*sig1*sig2*x[i]*y[j]*(uu[i+1, j+1] \
                            - uu[i+1, j-1] - uu[i-1, j+1] + uu[i-1, j-1])/(4*h*k) + uu[i, j]/dt

        u2[1:Nx+1, j] = TDMAsolver(b_x[1:],a_x, c_x[:-1], d_y)
        u22[1:Nx+1, j] = TDMAsolver(b_x[1:],a_x, c_x[:-1], d_yy)
    
    
    # Linear boundary condition
    u2[0,1:Ny+1] = 2*u2[1,1:Ny+1] - u2[2,1:Ny+1]
    u2[Nx+1, 1:Ny+1] = 2*u2[Nx, 1:Ny+1] - u2[Nx-1, 1:Ny+1]
    u2[0:Nx+2,0] = 2*u2[0:Nx+2,1] - u2[0:Nx+2,2]
    u2[0:Nx+2, Ny+1] = 2*u2[0:Nx+2, Ny] - u2[0:Nx+2, Ny-1]
    
    u22[0, 1:Ny+1] = 2*u22[1, 1:Ny+1] - u22[2, 1:Ny+1]
    u22[Nx+1, 1:Ny+1] = 2*u22[Nx, 1:Ny+1] - u22[Nx-1, 1:Ny+1]
    u22[0:Nx+2,0] = 2*u22[0:Nx+2,1] - u22[0:Nx+2,2]
    u22[0:Nx+2, Ny+1] = 2*u22[0:Nx+2, Ny] - u22[0:Nx+2,Ny-1]
    
    # y - direction 
    for i in range(1,Nx+1):
        
        for j in range(1,Ny+1):
            if method == 0: # ADI method
               d_x[j-1] = u2[i,j]/dt + ((sig1*x[i])**2)*(u2[i+1,j] - 2*u2[i,j] \
                          + u2[i-1,j])/(4*(h**2)) + (r-Dx)*x[i]*(u2[i+1,j] - u2[i,j])/(2*h)\
                          + (1/2)*rho*sig1*sig2*x[i]*y[j]*(u2[i+1,j+1] + u2[i-1,j-1] - u2[i-1,j+1] - u2[i+1,j-1])/(4*h*k)
                          
               d_xx[j-1] = u22[i,j]/dt + ((sig1*x[i])**2)*(u22[i+1,j] - 2*u22[i,j] + u22[i-1,j])/(4*(h**2)) \
                        + (r-Dx)*x[i]*(u22[i+1,j] - u22[i,j])/(2*h) + (1/2)*rho*sig1*sig2*x[i]*y[j]*(u22[i+1,j+1]\
                        + u22[i-1,j-1] - u22[i-1,j+1] - u22[i+1,j-1])/(4*h*k)
            elif method == 1: # OS method
            
                d_x[j-1] = (1/2)*rho*sig1*sig2*x[i]*y[j]*(u2[i+1, j+1]\
                           - u2[i+1, j-1] - u2[i-1, j+1] + u2[i-1, j-1])/(4*h*k) + u2[i, j]/dt
                d_xx[j-1] = (1/2)*rho*sig1*sig2*x[i]*y[j]*(u22[i+1, j+1]\
                           - u22[i+1, j-1] - u22[i-1, j+1] + u22[i-1, j-1])/(4*h*k) + u22[i, j]/dt
        
        u[i, 1:Ny+1] = TDMAsolver(b_y[1:], a_y, c_y[:-1], d_x)
        uu[i, 1:Ny+1] = TDMAsolver(b_y[1:], a_y, c_y[:-1], d_xx)
    
    # Linear boundary condition
    u[0,1:Ny+1] = 2*u[1,1:Ny+1] - u[2,1:Ny+1]
    u[Nx+1, 1:Ny+1] = 2*u[Nx, 1:Ny+1] - u[Nx-1, 1:Ny+1]
    u[0:Nx+2,0] = 2*u[0:Nx+2,1] - u[0:Nx+2,2]
    u[0:Nx+2, Ny+1] = 2*u[0:Nx+2, Ny] - u[0:Nx+2, Ny-1]
    
    uu[0, 1:Ny+1] = 2*uu[1, 1:Ny+1] - uu[2, 1:Ny+1]
    uu[Nx+1, 1:Ny+1] = 2*uu[Nx, 1:Ny+1] - uu[Nx-1, 1:Ny+1]
    uu[0:Nx+2,0] = 2*uu[0:Nx+2,1] - uu[0:Nx+2,2]
    uu[0:Nx+2, Ny+1] = 2*uu[0:Nx+2, Ny] - uu[0:Nx+2,Ny-1]
    
    # Early redemption 
    if np.mod(iteration,pp) == 0 and iteration < Nt:
        for i in range(Nx+2):
            for j in range(Ny+2):
                
                if min(x[i]/K0[0], y[j]/K0[1]) >= K[5 - int(np.floor(iteration/pp))]*F:
                    u[i, j] = (1 + c[5 - int(np.floor(iteration/pp))])*F
                    uu[i, j] = (1 + c[5 - int(np.floor(iteration/pp))])*F
#    iteration+=1                    
# Monte-Carlo simulation to compute the probatility that a knock-in occurs
# Monte-Carlo simulation is performed under P measure uisng mu1 and mu2
#%%
S1 = np.zeros(Nt); S2 = np.zeros(Nt);
S1[0] = K0[0]; S2[0] = K0[1];

count = 0
total_iter = 50000

while count < total_iter:
    for i in range(Nt-1):
        e1 = np.random.standard_normal(); e2 = np.random.standard_normal()
        S1[i+1]=S1[i]*np.exp((r-Dx-(sig1**2)/2)*dt+sig1*np.sqrt(dt)*e1)
        S2[i+1]=S2[i]*np.exp((r-Dy-(sig2**2)/2)*dt+sig2*np.sqrt(dt)*(rho*e1 + np.sqrt(1-rho**2)*e2))
    
    if sum(min(S1/S1[0], S2/S2[0]) < KI) > 0 :
           count+=1

#%%
# probability that a knock-in occurs
#prob = count/total_iter
Total_ELSPrice = (1-prob)*u + prob*uu
fdm_price = Total_ELSPrice[round(Nx*K0[0]/S1max)-1,round(Ny*K0[1]/S2max)-1]
print(fdm_price)

#%%

fig = plt.figure()
ax = fig.gca(projection='3d')
xnew,ynew = np.meshgrid(x,y)
surface = ax.plot_surface(xnew,ynew,Total_ELSPrice,cmap = cm.coolwarm)
plt.show()