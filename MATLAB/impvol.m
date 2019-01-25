function impvol = impvol(X,T)
C_P ='C';
S = 292.42;     % 기초자산 현재가격
r = 0.0165;
q = 0.017;
market_data = xlsread('ksp200call2.xlsx');
P = market_data(:,3);               % 시장가격
impvol = zeros(length(P),1);
for i = 1:length(X)
    impvol(i) =  NewtonRaphson_ImpVol(C_P,S,X(i),r,T(i),q,P(i));
end
impvol(~isfinite(impvol))= NaN;
impvol = fillmissing(impvol,'nearest');
end