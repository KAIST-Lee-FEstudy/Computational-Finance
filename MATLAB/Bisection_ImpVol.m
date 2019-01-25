%%%%%%%% Implied volatility(Bisection Method %%%%%%%%

function Imp_Vol = Bisection_ImpVol(C_P,S,X,r,T,q,MP)
% MP는 현재 옵션시장가격 

V_low = 0.01;                 % 변동성 초기값(lower boundary)
V_high = 2;                   % 변동성 초기값(upper boundary)
tol = 1e-6;                   % error tolerracne 

% 변동성 초기값 V_low일때 옵션가격 
P_low = BS_Vanilla(C_P,S,X,r,T,V_low,q);

% 변동성 초기값 V_high일때 옵션가격
P_high = BS_Vanilla(C_P,S,X,r,T,V_high,q);

vi = V_low + (MP-P_low)*(V_high-V_low)/(P_high-P_low);

% 내재변동성 Bisection 계산, 시장가와의 차이가 tol이하면 루프 중지 
while abs(MP-BS_Vanilla(C_P,S,X,r,T,vi,q)) > tol
    if BS_Vanilla(C_P,S,X,r,T,vi,q) < MP
        V_low = vi;
    else
        V_high = vi;
    end
    P_low = BS_Vanilla(C_P,S,X,r,T,V_low,q);
    P_high = BS_Vanilla(C_P,S,X,r,T,V_high,q);
    vi = V_low + (MP-P_low)*(V_high-V_low)/(P_high-P_low);
end
Imp_Vol = vi;
end
