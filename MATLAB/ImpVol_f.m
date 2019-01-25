%%%%%%%% Implied Volatility Function %%%%%%%%

function f = ImpVol_f(a,Mdata)

% Implied Volatility Function ¸ðµ¨ 

f = a(1)+a(3)*exp(a(2)*Mdata(:,2))+a(4)*Mdata(:,1)+a(5)*Mdata(:,1).^2+...
    a(6)*Mdata(:,1).^3+a(7)*Mdata(:,1).^4;
end
