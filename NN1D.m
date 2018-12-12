%{
Author: Arash Mehrjou
Generate 1D data, train a neural network and compute Taylor approximation
of the function approximated by the neural network.
%}
%%
close all
clear all
clc
syms x
%% Example 1
% delta = 1.0;
% f = @(x)-1 * min(max(x/delta, -1), 1);
% FUN = @(t,x)f(x);
%% Example 2
% f = - tanh(x);
% F = matlabFunction(f);
% F = @(t,X) F(X);
% G = @(t,X) 1/100;
%% Example 3
f = - tanh(x) + 0.1 * sin(5 * x);
F = matlabFunction(f);
F = @(t,X) F(X);
G = @(t,X) 1/100;
%% Create training data
NPERIODS = 200;
MTrajecs = 10;
NPERIODS = 300;
MTrajecs = 30;
experiment_num = 5;
timestep = 0.01;
[xtrain, ytrain] = createData(F, G, NPERIODS, MTrajecs, timestep);
dataset = [xtrain, ytrain];
%% Saving the dataset to replicate the experiment
% save(sprintf('data4exp%d.mat',experiment_num), 'dataset');
% load(sprintf('data4exp%d.mat',experiment_num));
%% Learning the NN from generated trajectories
clear net
Ts = 0.01;
data_in = dataset(:,1)';
data_out = dataset(:,2)';
nhiddens = 15;
net = feedforwardnet([nhiddens]);
net = train(net,data_in,data_out);
genFunction(net, 'netfcn','MatrixOnly','yes')
syms x
f_learned = @(x)netfcn(x);
%% Plotting the true function and the learned function by the NN
x_samples = -10:0.01:10;
y_samples = f_learned(x_samples);
plot(x_samples,y_samples, 'k')
hold on
f_temp = matlabFunction(f);
plot(x_samples, f_temp(x_samples),'r')
grid on
%% Computing Taylor expansion
taylor_order = 4;
x_expansion = -3; % expansion point
netfcn_replacer(1) % modifying the generated file for 1 dimeniosnal function to make it differentiable
f_learned_symbolic = f_learned(x);
disp('Computing Taylor expansion...')
f_taylor_symbolic = taylor(f_learned_symbolic, x, 'Order', taylor_order, 'ExpansionPoint', x_expansion);
f_taylor = matlabFunction(f_taylor_symbolic);
x_samples = x_expansion-1:0.01:x_expansion+1;
y_samples = f_taylor(x_samples);
hold on
plot(x_samples,y_samples, 'b')

%% formal plot: True function vs NN-learned function
disp('Plotting...')
syms x
f_learned = @(x)netfcn(x);
% x_samples = -20:0.2:20;
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
p = plot(x_samples, f_temp(x_samples),'-')
set(p,'Color',[0.4 0.7 0.4],'LineWidth',1.5)
hold on
p = plot(x_samples,y_samples, '--')
set(p,'Color','red','LineWidth',2)
grid on
% xticks_values = linspace(x_samples(1),x_samples(end),11);
xticks_values = linspace(-5.24,4.96,11);
set(gca,'XTick',xticks_values)
xTickLabel = arrayfun(@(x)sprintf('%3.2f',x),xticks_values, 'UniformOutput', false);
set(gca,'XTickLabel',xTickLabel)
% xlim([x_samples(1),x_samples(end)])
ylim([min([f_temp(x_samples), y_samples])-0.5,max([f_temp(x_samples), y_samples])+0.5])
yticks_values = linspace(-2,2,5);
set(gca,'YTick',yticks_values)
% set(gca,'XTick',linspace(x_samples(1),x_samples(end),11))
set(gca,'XTick',linspace(-5.24,4.96,11))
% set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),11),2))
xlabel('$x$', 'Interpreter','latex', 'fontsize', 14)
ylabel('$y$', 'Interpreter','latex', 'fontsize', 14)
leg1 = legend('$f$','$\hat{f}$');
set(leg1, 'Interpreter','latex');
a=get(leg1,'position')
set(leg1,'position',[0.8032, 0.7, 0.1211, 0.1509])
set(leg1,'Interpreter','latex')
set(leg1,'FontSize', 10)
% title('Simulation Results')
grid on
% latex_fig(5, 3, 1.5)
% filename = sprintf('f_true_learned_[%d].pdf', nhiddens);
filename = sprintf('f_true_learned_[10_5].pdf');
print(filename, '-dpdf');
movefile(filename,'./figs') 
%% formal plot: NN-learned function vs Taylor approximated local function
x_for_learned = linspace(-10,10,1000);
x_for_taylor = linspace(x_expansion-1.5, x_expansion+1.5, 100);
FigW=13.49414;
FigH=21.08462*.3;
Figure1=figure(1);clf;
    set(Figure1,'defaulttextinterpreter','latex',...
                'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                'Position',[1,10,FigW,FigH]);
subaxis(1,1,1,'ml',0.1, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
p = plot(x_for_taylor,f_taylor(x_for_taylor),'.-')
set(p,'Color','red','LineWidth',2)
hold on
p = plot(x_for_learned,f_learned(x_for_learned),'-')
set(p,'Color','black','LineWidth',1.5)

set(gca,'XTick',linspace(x_for_learned(1),x_for_learned(end),11))
% set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),11),2))
xlabel('$x$', 'Interpreter','latex', 'fontsize', 10)
ylabel('$y$', 'Interpreter','latex', 'fontsize', 10)
leg1 = legend('$f$','$\hat{f}$');
set(leg1, 'Interpreter','latex');
a=get(leg1,'position')
set(leg1,'position',[a(1), a(2)/3, a(3), a(4)])

set(leg1,'Interpreter','latex')
set(leg1,'FontSize', 10)
% title('Simulation Results')
grid on
% latex_fig(5, 3, 1.5)
filename = sprintf('f_true_taylor.pdf');
print(filename, '-dpdf');
movefile(filename,'./figs') 

