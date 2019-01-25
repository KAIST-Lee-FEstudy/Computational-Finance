%%%%%%%%%%% SMA log-likelihood function %%%%%%%%%%%

function f = SMA_llh(b,x)
f = sum(log(b)+(x.^2)/b);
end

