%%%%%%%% NewtonRaphson_ImpVol %%%%%%%%
function ImpVol = NewtonRaphson_ImpVol(C_P,S,X,r,T,q,MP)

vol =0.1;               % ������ �ʱⰪ
tol = 1e-6;             % ���� �Ѱ� 

% ������ �ʱⰪ�� ���� �ɼǰ���

P = BS_Vanilla(C_P,S,X,r,T,vol,q);

% ������ �ʱⰪ�� ���� ���� 

vega = BS_Vanilla_Greeks(C_P,'vega',S,X,r,T,vol,q);

% ���纯���� ��� ,Newton-Raphson method
while abs(P-MP)>tol
    vol = vol - (P-MP)/vega;
    P = BS_Vanilla(C_P,S,X,r,T,vol,q);
    vega = BS_Vanilla_Greeks(C_P,'vega',S,X,r,T,vol,q);
end
ImpVol = vol;
end


