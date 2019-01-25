%% BoxMullerMarsaglia
close all;
tic;
% RNG('twister',5489)

N = 1e4; N25 = N*2.5;
U1 = 2*rand(1,N25)-1;             % U1, U2 generate, 사각형 안에서의 Uniform generate
U2 = 2*rand(1,N25)-1;             % U1, U2 generate, 사각형 안에서의 Uniform generate
w = U1.^2 + U2.^2;                % Acceptance, Rejection method 
wIndex = find(w<=1);              % If w =< 1 :accpet, else : reject 
wIndex = wIndex(1:N);
ww =w(wIndex);
UU1 = U1(wIndex);
UU2 = U2(wIndex);
c = sqrt(-2*log(ww)./ww);
z1 = c.*UU1;
z2 = c.*UU2;
yran = [z1 z2];
N2 = 2*N;

% Goodness-of-Fit Test by Lilliefors
[H,P_value,stat,CV] = kstest(yran,[],0.05)

% Plotting Histogram and True PDF
xnum = -3.9:0.2:3.9;
N2height = hist(yran,xnum)/0.2/N2;
bar(xnum,N2height,1,'w')
set(gca,'fontsize',11,'fontweight','bold')
hold on
xx = linspace(-4,4,201);
pp = normpdf(xx);
plot(xx,pp,'r','linewidth',2)
legend('Histogram','True PDF')
xlabel('x'),ylabel('Relative Frequency')
axis([-4 4 0 0.65])
hold off
saveas(gcf ,'BoxMullerMarsaglia101 ','epsc ')
save('BoxMullerMarsaglia101 ','yran')

wholetime = toc




