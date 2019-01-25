%%%%%%%% NewtonRaphson_ImpVol %%%%%%%%
function ImpVol = NewtonRaphson_ImpVol(C_P,S,X,r,T,q,MP)

vol =0.1;               % 변동성 초기값
tol = 1e-6;             % 오차 한계 

% 변동성 초기값에 대한 옵션가격

P = BS_Vanilla(C_P,S,X,r,T,vol,q);

% 변동성 초기값에 대한 베가 

vega = BS_Vanilla_Greeks(C_P,'vega',S,X,r,T,vol,q);

% 내재변동성 계산 ,Newton-Raphson method
while abs(P-MP)>tol
    vol = vol - (P-MP)/vega;
    P = BS_Vanilla(C_P,S,X,r,T,vol,q);
    vega = BS_Vanilla_Greeks(C_P,'vega',S,X,r,T,vol,q);
end
ImpVol = vol;
end


