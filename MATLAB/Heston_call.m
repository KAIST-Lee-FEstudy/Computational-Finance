S=100;
K=100;
T=30/365;
r=0;
v=0.01;
kappa=2;
theta=0.01;
lambda=0;
sigma=0.1;
rho=0;
call=1.1345;
N=100;
M=zeros(1,10);
c1=zeros(1,10);
c2=zeros(1,10);
for i=1:10
M(i)=1000*i;
[price,err]=Heston_MCS(S,K,T,r,v,kappa,theta,lambda,sigma,rho,N,M(i));
c1(i)=price-1.96*err;
c2(i)=price+1.96*err;
fprintf('Standard Error : %.4f\n', err)
end
plot(M,ones(1,10)*call);
hold on;
plot(M,c1);
hold on;
plot(M,c2);
% fprintf('Standard Error : %.4f\n', err)