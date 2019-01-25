function [price err] = Heston_MCS(S,K,T,r,v,kappa,theta,lambda,sigma,rho,N,M)

kappa_s=kappa+lambda;
theta_s=kappa*theta/(kappa+lambda);

dt=T/N;


C=zeros(M,1);   

for j=1:M
S_m=zeros(N+1,1);
v_m=zeros(N+1,1);
S_m(1)=S;
v_m(1)=v;

for i=1:N
    
    e1=norminv(random('unif',0,1),0,1);
    e2_temp=norminv(random('unif',0,1),0,1);
    e2=e1*rho+e2_temp*sqrt(1-rho*rho);
    S_m(i+1)=S_m(i)*exp((r-0.5*max(v_m(i),0))*dt+sqrt(max(v_m(i),0))*sqrt(dt)*e1);
    v_m(i+1)=v_m(i)+kappa_s*(theta_s-max(v_m(i),0))*dt+sigma*sqrt(max(v_m(i),0))*sqrt(dt)*e2;
    
end

C(j)=exp(-r*T)*max(S_m(N+1)-K,0);
end

price=mean(C);
err=std(C)/sqrt(M);
