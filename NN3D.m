%{
Author: Arash Mehrjou
Generate 3D data, train a neural network and compute Taylor approximation
of the function approximated by the neural network.
%}
%%
close all
clear all
clc

%% Example 1
syms x y z
fvars = [x y z];
sigma = 10;
beta = 8/3;
rho = 30;
f = [sigma * (y - x);
    x * (rho - z) - y ;
    x * y - beta * z ];
f_true = matlabFunction(f);
f_true = @(t,x)f_true(x(1),x(2),x(3));
FUN = matlabFunction(f);
FUN = @(t,x)FUN(x(1),x(2), x(3));
%% Create training data
dT = 0.01;
disp('Dataset is being generated...')
for x0=-10:0.02:10
    [t,xexact] = ode45(FUN, [0:dT:1], [x0;x0; x0]);
    x = xexact(1:end-1, :);
    x_delayed = xexact(2:end, :);
    dx = x_delayed - x;
    dxdt = dx/dT;
    if size(dataset,1)==0
        dataset = cat(2, x, dxdt);
    else
        dataset = cat(1, dataset, cat(2, x, dxdt));
    end
end
%% Learning the NN from generated trajectories
clear net
Ts = 0.01;
data_in = dataset(:,1:3)';
data_out = dataset(:,4:6)';
net = feedforwardnet([50]);
net = train(net,data_in,data_out);
genFunction(net, 'netfcn','MatrixOnly','yes')
netfcn_replacer(3)
x = sym('x', [3,1]);
f_learned = @(x)netfcn(x);
%% Plotting the true function and the learned function by the NN
x_samples = -10:0.01:10;
y_samples = f_learned([x_samples; 1*x_samples; 1*x_samples]);
plot(x_samples,y_samples, 'k');
hold on
f_temp = matlabFunction(f);
f_temp = @(x)f_temp(x(1,:), x(2,:), x(3, :));
plot(x_samples, f_temp([x_samples; 1*x_samples; 1*x_samples]),'r')
grid on
%% Computing Taylor expansion
x_expansion = [-2,1,1];
taylor_order = 3;
x = sym('x', [3,1]);
f_learned_symbolic = netfcn(x);
disp('Computing Taylor expansion...')
f_taylor_symbolic = taylor(f_learned_symbolic, x, 'Order', taylor_order, 'ExpansionPoint', x_expansion);
f_taylor = matlabFunction(f_taylor_symbolic);
f_taylor = @(x)f_taylor(x(1,:),x(2,:),x(3,:));
x_samples = x_expansion(1)-1:0.01:x_expansion(1)+1;
y_samples = f_taylor([x_samples; 1*x_samples; 1*x_samples]);
hold on
plot(x_samples,y_samples, 'b');
%% formal plot: True function vs NN-learned function
disp('Plotting...')
syms x
f_learned = @(x)netfcn(x);
x_samples = min(dataset(:,1)):0.2:max(dataset(:,1))+1;
y_samples = f_learned(x_samples);
FigW=13.49414;
FigH=21.08462*.3;
Figure1=figure(1);clf;
    set(Figure1,'defaulttextinterpreter','latex',...
                'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                'Position',[1,10,FigW,FigH]);
subaxis(1,1,1,'ml',0.1, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
f_temp = matlabFunction(f);
f_temp = @(x)f_temp(x(1,:), x(2,:), x(3,:));
p1 = plot(x_samples, f_temp([x_samples; 1*x_samples; 1*x_samples]),'-');
set(p1,'Color',[0.4 0.7 0.4],'LineWidth',1.5)
hold on
p2 = plot(x_samples,y_samples, '--');
set(p2,'Color','red','LineWidth',2)
grid on
set(gca,'XTick',linspace(x_samples(1),x_samples(end),11))
xlabel('$x$', 'Interpreter','latex', 'fontsize', 10)
ylabel('$y$', 'Interpreter','latex', 'fontsize', 10)
leg1 = legend([p1(1), p2(1)],'$f$','$\hat{f}$');
set(leg1, 'Interpreter','latex');
a=get(leg1,'position');
% set(leg1,'position',[0.8032, 0.5, 0.1211, 0.1509])
set(leg1,'Interpreter','latex')
set(leg1,'FontSize', 10)
% title('Simulation Results')
ylim([-110,200]);
grid on
% latex_fig(5, 3, 1.5)
filename = sprintf('f_true_learned3d.pdf');
print(filename, '-dpdf');
movefile(filename,'./figs') 
