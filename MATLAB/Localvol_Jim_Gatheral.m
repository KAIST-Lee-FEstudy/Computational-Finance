% Calculation of local volatility from implied volatility using the mapping
% described in Jim Gatheral's book.  Compares the results with local
% volatility obtained from a vendor product.

clc; clear;

% Input the maturities in days and the strikes
T = [  3  31  53  88 144 452]';
K = [450 460 470 480 490 500 510 520];

% Input the vendor implied volatility.
IV = [...
   0.27782000000000   0.26289000000000   0.25015000000000   0.23979000000000   0.23192000000000   0.22665000000000   0.22408000000000   0.22429000000000;
   0.28608000000000   0.27624000000000   0.26704000000000   0.25853000000000   0.25081000000000   0.24404000000000   0.23835000000000   0.23386000000000;
   0.30739000000000   0.30039000000000   0.29380000000000   0.28763000000000   0.28188000000000   0.27655000000000   0.27166000000000   0.26720000000000;
   0.30986000000000   0.30412000000000   0.29870000000000   0.29359000000000   0.28878000000000   0.28426000000000   0.28003000000000   0.27608000000000;
   0.31581000000000   0.31039000000000   0.30532000000000   0.30058000000000   0.29617000000000   0.29208000000000   0.28828000000000   0.28477000000000;
   0.34581000000000   0.34302000000000   0.34032000000000   0.33771000000000   0.33519000000000   0.33274000000000   0.33037000000000   0.32806000000000];

% Input the vendor local volatility.
VendorLV = [...
   0.27606806760295   0.31509608735873   0.28521918421414   0.26468152197409   0.24736542961237   0.23817659292558   0.23514260184238   0.23684459024326;
   0.38919808572039   0.35400754222292   0.34064999429828   0.32398496861290   0.30951212704022   0.29850889463738   0.28702200347046   0.28879597480659;
   0.34930411868940   0.32015999864869   0.31232617961846   0.30087891926809   0.29171530998974   0.28293874743742   0.27475307414097   0.27162389017082;
   0.35922980554791   0.32520685839555   0.31961371978572   0.30851232551092   0.30021190053790   0.29217680356219   0.28473514459900   0.28114485871059;
   0.40334383654753   0.35313999159861   0.35540230642065   0.34472758243737   0.33832748111828   0.33422049594074   0.32597578997920   0.32754288381047;
   0.37524309175323   0.33089708530821   0.33756909219412   0.32998579422628   0.33463277404219   0.32486902768903   0.33267730681432   0.31510958178130];
[R, C] = size(IV);

% Annualized Maturity.
Basis = 360;
T = T./Basis;

% Allocation
w = zeros(R,C);
dwdT = zeros(R,C);
dwdy = zeros(R,C);
dw2dy2 = zeros(R,C);
LV = zeros(R,C);
error = zeros(R,C);

% Construct total implied variance.
for j=1:C
   w(:,j) = IV(:,j).^2 .* T;
end

% Forwards prices.
S = 484.81;
r = 0.0;
q = 0.0;
y = log(K./S);

% Increments for finite-difference derivatives.
dt = 10e-6;
dy = 1e-10;

% First derivatives with respect to time.
% Calculate backward, forward, and central differences, but retain central.
for j=1:C
   for i=1:R
      T0 = T(i) - dt;
      T1 = T(i);
      T2 = T(i) + dt;
        w0 = interp1(T,w(:,j),T0,'spline');
      w1 = w(i,j);
      w2 = interp1(T,w(:,j),T2,'spline');
      Forward   = (w2-w1)/(T2-T1);
      Backward  = (w1-w0)/(T1-T0);
      Central   = (w2-w0)/(T2-T0);
      dwdT(i,j) = Backward; % Choose any scheme among Forward, Backward, Central 
   end
end

% First derivatives with respect to moneyness.
for i=1:R
   for j=1:C
      y0 = y(j) - dy;
      y1 = y(j);
      y2 = y(j) + dy;
      w0 = interp1(y,w(i,:),y0,'spline');
      w1 = w(i,j);
      w2 = interp1(y,w(i,:),y2,'spline');
      Forward   = (w2-w1)/(y2-y1);
      Backward  = (w1-w0)/(y1-y0);
      Central   = (w2-w0)/(y2-y0);
      dwdy(i,j) = Central; % Choose any scheme among Forward, Backward, Central
   end
end

% Second derivative with respect to moneyness.
for i=1:R
   for j=1:C
      y0 = y(j) - dy;
      y1 = y(j);
      y2 = y(j) + dy;
      w0 = interp1(y,dwdy(i,:),y0,'spline');
      w1 = w(i,j);
      w2 = interp1(y,dwdy(i,:),y2,'spline');
      Forward   = (w2-w1)/(y2-y1);
      Backward  = (w1-w0)/(y1-y0);
      Central   = (w2-w0)/(y2-y0);
      dw2dy2(i,j) = Central; % Choose any scheme among Forward, Backward, Central
   end
end

% Mapping of implied vol to local vol using the formula in Jim Gatheral's
% book.
for i=1:R
   for j=1:C
      num = dwdT(i,j);
      den = 1 - y(j)/w(i,j)*dwdy(i,j) ...
         + 0.25*(-0.25 - 1/w(i,j) + y(j)^2/w(i,j)^2)*(dwdy(i,j))^2 ...
          + 0.5*dw2dy2(i,j);
      LV(i,j) = sqrt(num/den);
      %error(i,j) = (LV(i,j) - VendorLV(i,j))^2;
        error(i,j) = (LV(i,j) - IV(i,j))^2;
   end
end

% Display the vendor local vol and the calculated local vol.
%VendorLV
%LV

% Plot them.
surf(LV);
hold on
%surf(VendorLV);
%hold off

% Calculate the error between the vendor estimates and these estimates.
Error1 = mean(error(i,j))
Error2 = sqrt(mean(error(i,j)))
