%%%%%%%% Implied volatility(Bisection Method %%%%%%%%

function Imp_Vol = Bisection_ImpVol(C_P,S,X,r,T,q,MP)
% MP�� ���� �ɼǽ��尡�� 

V_low = 0.01;                 % ������ �ʱⰪ(lower boundary)
V_high = 2;                   % ������ �ʱⰪ(upper boundary)
tol = 1e-6;                   % error tolerracne 

% ������ �ʱⰪ V_low�϶� �ɼǰ��� 
P_low = BS_Vanilla(C_P,S,X,r,T,V_low,q);

% ������ �ʱⰪ V_high�϶� �ɼǰ���
P_high = BS_Vanilla(C_P,S,X,r,T,V_high,q);

vi = V_low + (MP-P_low)*(V_high-V_low)/(P_high-P_low);

% ���纯���� Bisection ���, ���尡���� ���̰� tol���ϸ� ���� ���� 
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
