%%%%%%%% European_Vanilla option Greeks(Analytic) %%%%%%%%

function result = BS_Vanilla_Greeks(C_P,Greeks, S,X,r,T,vol,q)

% d1, d2 계산 

d1 = (log(S./X)+(r-q+0.5*vol.^2).*T)./(vol.*sqrt(T));
d2 = d1 - vol.*sqrt(T);

% 옵션가격 및 민감도 계산
% (N'(d1) : Normal probability density function , mean =0 , sigma =1)

if C_P == 'C'
    switch Greeks
        case 'price'
            result = S.*exp(-q.*T).*normcdf(d1)-X.*exp(-r.*T).*normcdf(d2);
        case 'delta'
            result = exp(-q.*T).*normcdf(d1);
        case 'gamma'
            result = (normpdf(d1).*exp(-q*T))./(S.*vol.*sqrt(T));
        case 'vega'
            result = S.*sqrt(T).*normpdf(d1).*exp(-q.*T);
        case 'theta'
            result = (-S.*normpdf(d1).*vol.*exp(-q*T))./...
                     (2.*sqrt(T))+q.*S.*normcdf(d1).*exp(-q.*T)-r.*X.*...
                     exp(-r.*T).*normcdf(d2);
        case 'rho'
            result = normcdf(d2).*X.*T.*exp(-r.*T);
        case 'psi'
            result = -S.*T.*exp(-q.*T).*normcdf(d1);
    end
    
elseif C_P == 'P'
    switch Greeks
        case 'price'
            result = X.*exp(-r.*T).*normcdf(-d2)-S.*exp(-q.*T).*normcdf(-d1);
        case 'delta'
            result = -exp(- q.*T).*normcdf(-d1);
        case 'gamma'
            result = (normpdf(d1).*exp(-q*T))./(S.*vol.*sqrt(T));
        case 'vega'
            result = S.*sqrt(T).*normpdf(d1).*exp(-q.*T);
        case 'theta' 
            result = (-S.*normpdf(d1).*vol.*exp(-q*T))./...
                     (2.*sqrt(T))-q.*S.*normcdf(-d1).*exp(-q.*T)+r.*X.*...
                     exp(-r.*T).*normcdf(-d2);
        case 'rho'
            result = normcdf(-d2).*-(X.*T.*exp(-r.*T));
        case 'psi'
            result = S.*T.*exp(-q.*T).*normcdf(-d1);
    end
end

            