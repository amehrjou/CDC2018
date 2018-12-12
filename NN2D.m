%{
Author: Arash Mehrjou
Generate 2D data, train a neural network and compute Taylor approximation
of the function approximated by the neural network.
%}
%%
close all
clear all
clc
x = sym('x', [2,1]);
%% Example 1
x0 = [pi/2;0];
r = 1;
g = 9.8;
l = 0.1;
f = [x(2); 
     - r * x(2) - g / l * sin(x(1))];
FUN = matlabFunction(f);
FUN = @(t,x)FUN(x(1),x(2));
%% Create training data
dT = 0.01;
disp('Dataset is being generated...')
for x0=-5:0.01:5
    [t,xexact] = ode45(FUN, [0:dT:1], [x0;x0]);
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
data_in = dataset(:,1:2)';
data_out = dataset(:,3:4)';
net = feedforwardnet([10, 10]);
net = train(net,data_in,data_out);
genFunction(net, 'netfcn','MatrixOnly','yes')
netfcn_replacer(2)
x = sym('x', [2,1]);
f_learned = @(x)netfcn(x);
%% Plotting the true function and the learned function by the NN
x_samples = -10:0.01:10;
y_samples = f_learned([x_samples; 1*x_samples]);
plot(x_samples,y_samples, 'k');
hold on
f_temp = matlabFunction(f);
f_temp = @(x)f_temp(x(1,:), x(2,:));
plot(x_samples, f_temp([x_samples; 1*x_samples]),'r')
grid on
%% Computing Taylor expansion
x_expansion = [-2,1];
taylor_order = 3;
x = sym('x', [2,1]);
f_learned_symbolic = netfcn(x);
disp('Computing Taylor expansion...')
f_taylor_symbolic = taylor(f_learned_symbolic, x, 'Order', taylor_order, 'ExpansionPoint', x_expansion);
f_taylor = matlabFunction(f_taylor_symbolic);
f_taylor = @(x)f_taylor(x(1,:),x(2,:));
x_samples = x_expansion(1)-1:0.01:x_expansion(1)+1;
y_samples = f_taylor([x_samples; 1*x_samples]);
hold on
plot(x_samples,y_samples, 'b');
%% formal plot: True function vs NN-learned function
disp('Plotting...')
x = sym('x', [2,1]);
f_learned = @(x)netfcn(x);
x_samples = -10:0.01:10;
y_samples = f_learned([x_samples; 1*x_samples]);
FigW=13.49414;
FigH=21.08462*.3;
Figure1=figure(1);clf;
    set(Figure1,'defaulttextinterpreter','latex',...
                'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                'Position',[1,10,FigW,FigH]);
subaxis(1,1,1,'ml',0.1, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
f_temp = matlabFunction(f);
f_temp = @(x)f_temp(x(1,:), x(2,:));
p1 = plot(x_samples, f_temp([x_samples; 1*x_samples]))
set(p1,'Color',[0.2 0.7 0.2],'LineWidth',2)
hold on
p2 = plot(x_samples,y_samples, '--')
set(p2,'Color','red','LineWidth',2)
grid on
set(gca,'XTick',linspace(x_samples(1),x_samples(end),11))
% set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),11),2))
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
filename = sprintf('f_true_learned2d.pdf');
print(filename, '-dpdf');
movefile(filename,'./figs') 
%% formal plot: NN-learned function vs Taylor approximated local function
x_for_learned = linspace(-10,10,1000);
x_for_taylor = x_expansion(1)-5:0.01:x_expansion(1)+5;
FigW=13.49414;
FigH=21.08462*.3;
Figure1=figure(1);clf;
    set(Figure1,'defaulttextinterpreter','latex',...
                'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                'Position',[1,10,FigW,FigH]);
subaxis(1,1,1,'ml',0.1, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
p1 = plot(x_for_learned,f_learned(x_for_learned),'--');
set(p1,'Color','red','LineWidth',2);
hold on
p2 = plot(x_for_taylor,f_taylor([x_for_taylor; x_for_taylor]),'-');
set(p2,'Color','blue','LineWidth',2);
set(gca,'XTick',linspace(x_for_learned(1),x_for_learned(end),11))
% set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),11),2))
xlabel('$x$', 'Interpreter','latex', 'fontsize', 10)
ylabel('$y$', 'Interpreter','latex', 'fontsize', 10)
legend('a','b','c');
leg1 = legend([p1(1),p2(1)],'$\hat{f}$','$\tilde{f}$');
set(leg1, 'Interpreter','latex');
a=get(leg1,'position');
% set(leg1,'position',[a(1), a(2)/3, a(3), a(4)])
ylim([-110,200]);
set(leg1,'FontSize', 10)
% title('Simulation Results')
grid on
% latex_fig(5, 3, 1.5)
filename = sprintf('f_true_taylor2d.pdf');
print(filename, '-dpdf');
movefile(filename,'./figs') 

