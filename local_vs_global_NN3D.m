%{
Author: Arash Mehrjou
This script compute the total representation cost of a system when it is
considered globally or as a concatentaion of some local approximations.
%}
clc
clear all
close all
x = sym('x', [3,1]);
%% Experiment 1 (Load the learned function and generate data)
fvars = [x(1), x(2), x(3)];
f = netfcn(x);
f_true = @(t, x)netfcn(x);
% racalling the original function for the plots
sigma = 10;
beta = 8/3;
rho = 30;
f_orig = [sigma * (x(2) - x(1));
    x(1) * (rho - x(3)) - x(2);
    x(1) * x(2) - beta * x(3)];
f_orig = matlabFunction(f_orig);
f_orig = @(t,x)f_orig(x(1),x(2),x(3));
x0 = [-0.6, 0.4, 0.1];
x0 = [-2, -2, -2];
timestep = 0.01;
lambda = 0.2; % Relative weight of the number of coefficients.
T_total = 0.4; % Total time on which the global approximation is computed
T_nsections_min = 1;
T_nsections_max = 3;
min_k = 2;
max_k = 2;
T_nsections = 1; % This determines each local T horizons
tspan = 0:timestep:T_total; % duration of each local time horizon
[t,xtrue] = ode45(f_true, [0:timestep:T_total], x0 );
[t,xorigfull] = ode45(f_orig, [0:timestep:T_total], x0 );
%% Compute local cost for different number of local sections
k_total_local_save = {};
L_total_local_save = {};
xhat_total_local_save = {};
xorig_total_local_save = {};
ind = 1;
all_k_dict = {};
fprintf('Computing local approximations cost ...\n')
for T_nsections=T_nsections_min:1:T_nsections_max
    T_section = T_total / T_nsections;
    L_total_local = 0;
    k_total_local = 0;
    x0_local = x0;
    [indspan, tspan] = last_indices(0, T_total, timestep, T_nsections);
    xhat_total=[0,0,0];
    xorig_total=[0,0,0];
    all_k_dict{T_nsections} = [];
    for n=1:1:T_nsections
        [k, Lopt, xhat, xorig] = choosek_NN3D(tspan{n}, f, f_orig, fvars, x0_local, timestep, lambda, min_k, max_k);
        all_k_dict{T_nsections} = [all_k_dict{T_nsections}, k];
        xhat_total = cat(1, xhat_total, xhat);
        xorig_total = cat(1, xorig_total, xorig);        
        if n ~= T_nsections
            x0_local = xtrue(indspan{n+1}(1),:);
        end
        L_total_local = L_total_local + Lopt;
        k_total_local = k_total_local + k;
    end
    k_total_local_save{ind} = k_total_local;
    L_total_local_save{ind} = L_total_local;
    xhat_total_local_save{ind} = xhat_total(2:end,:);
    xorig_total_local_save{ind} = xorig_total(2:end,:);
    ind = ind + 1;
end
k_total_local_save = cell2mat(k_total_local_save);
L_total_local_save = cell2mat(L_total_local_save);

%% Compute global cost (1 local section)
fprintf('Computing global approximation cost ...')
T_nsections = 1;
T_section = T_total / T_nsections;
tspan = 0:timestep:T_total;
L_total_global = 0;
k_total_global = 0;
x0_global = x0;
[k, Lopt, xT, ~] = choosek_NN3D(tspan, f, f_orig, fvars, x0_global, timestep, lambda, min_k, max_k);
disp(x0_global)
x0_global = xT;
L_total_global = L_total_global + Lopt;
k_total_global = k_total_global + k;

%% Plotting true trajectory and trajectory obtained by local approximations
state_characters = {'x_1', 'x_2', 'x_3'};
for state_to_plot=1:1:length(fvars)
    max_hat = -1000;
    min_hat = 1000;
    for m=T_nsections_min:T_nsections_max
        FigW=13.49414;
        FigH=21.08462*.3;
        Figure1=figure(1);clf;
            set(Figure1,'defaulttextinterpreter','latex',...
                        'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                        'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                        'Position',[1,10,FigW,FigH]);
        subaxis(1,1,1,'ml',0.16, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
        p1 = plot(t,xorigfull(:,state_to_plot))
        hold on
        p2 = plot(t,xtrue(:,state_to_plot))
        hold on
        set(p1,'LineWidth',2, 'Color','green')
        set(p2,'LineWidth',2, 'Color','red')
        p = plot(t , xhat_total_local_save{m-T_nsections_min+1}(:,state_to_plot), '--');
        max_hat = max(max(xhat_total_local_save{m-T_nsections_min+1}(:,state_to_plot)), max_hat);
        min_hat = min(min(xhat_total_local_save{m-T_nsections_min+1}(:,state_to_plot)), min_hat);
        set(p,'LineWidth',2, 'Color','blue')
        %%coloring the background
        %ylim( [1 3] )
        set(gca,'XTick',linspace(tspan(1),tspan(end),10), 'fontsize', 11.5)
        set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),10),1), 'fontsize', 11.5)
        xlabel('${\rm time}(t)$', 'Interpreter','latex', 'fontsize', 16)
        ylabel(sprintf('$%s(t)$',state_characters{state_to_plot}), 'Interpreter','latex', 'fontsize', 16)
        % title('Simulation Results')
        %set(p,'Color','red','LineWidth',2)
        set(p,'LineWidth',2)
        grid on
        % latex_fig(5, 3, 1.5)
        xlim([0 max(t)]);
        yl = [min(min_hat, min(xtrue(:,state_to_plot))), max(max_hat, max(xtrue(:,state_to_plot)))];
        yl = [-100,100];
        ylim(yl);
        T_length = T_total/m;
        for mm=1:1:m
            start_x = (mm-1) * T_length;
            stop_x = mm * T_length;
            if mod(mm,2)==0
                fill([start_x start_x stop_x stop_x],[yl(1) yl(2) yl(2) yl(1)],'black')
                alpha(0.05)
            else
                if mm ~=m
                    fill([start_x start_x stop_x stop_x],[yl(1) yl(2) yl(2) yl(1)],'white')
                    alpha(0.05)
                end
            end
            text_x = (start_x + stop_x) / 2;
            text_y = (yl(2) + yl(1)) / 2 - 50;
%             if state_to_plot == 2
%                 text_y = yl(1)+1;
%             end
            text(text_x, text_y, sprintf('$k_%d=%d$',mm,all_k_dict{m}(mm)), 'fontsize', 13, 'Interpreter', 'latex');
            
        end
%         legend({'$x(t)$', '$\hat{x}(t)$', '$\tilde{x}(t)$' },'Interpreter','latex','FontSize',10)
        leg1 = legend({'$x(t)$', '$\hat{x}(t)$', '$\tilde{x}(t)$'});
        set(leg1, 'Interpreter','latex');
        a=get(leg1,'position');
        set(leg1,'position',[0.2, 0.7, 0.1211, 0.1509]);
        set(leg1,'Interpreter','latex');
        set(leg1,'FontSize', 10);
        filename = sprintf('ms_nn_part_%d_state_%d_3D.pdf',m, state_to_plot);
        print(filename, '-dpdf');
        movefile(filename,'./figs') 
    end
end
close all
Figure1=figure(1);clf;
    set(Figure1,'defaulttextinterpreter','latex',...
                'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                'Position',[1,10,FigW,FigH]);
subaxis(1,1,1,'ml',0.12, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
state_to_plot = 2;
p = plot(T_nsections_min:1:T_nsections_max, L_total_local_save);
set(p,'LineWidth',2, 'Color','blue')
hold on
% p = plot(T_nsections_min:1:T_nsections_max, L_total_global*ones(size(T_nsections_min:1:T_nsections_max)))
% set(p,'LineWidth',2, 'Color','red')
% ylim( [1 3] )
set(gca,'XTick',T_nsections_min:T_nsections_max, 'fontsize', 12)
set(gca,'XTickLabel',T_nsections_min:T_nsections_max, 'fontsize', 12)
xlabel('$m$', 'Interpreter','latex', 'fontsize', 14)
ylabel('$L_{\rm total}(m)$', 'Interpreter','latex', 'fontsize', 14)
% title('Simulation Results')
%set(p,'Color','red','LineWidth',2)
set(p,'LineWidth',2)
grid on
% latex_fig(5, 3, 1.5)
filename = sprintf('cost_vs_nsections_3D.pdf',m, state_to_plot);
print(filename, '-dpdf');
movefile(filename,'./figs') 


%% Plotting Taylors approximations over the learned function
% xmin = min(xhat_total_local_save{1});
% xmax = max(xhat_total_local_save{1});
xmin = -100;
xmax =100;
syms x
f_temp_symbolic = netfcn(x);
f_temp = matlabFunction(f_temp_symbolic);
dx = 0.01;
ylims = [-1.5,1.5] * 60;
% x_nsections_min=1;
% x_nsections_max=10;
T_nsections_min = 2;
for m=T_nsections_min:T_nsections_max
    FigW=13.49414;
    FigH=21.08462*.3;
    Figure1=figure();clf;
        set(Figure1,'defaulttextinterpreter','latex',...
                    'PaperUnits','centimeters','PaperSize',[FigW FigH],...
                    'PaperPosition',[0,0,FigW,FigH],'Units','centimeters',...
                    'Position',[1,10,FigW,FigH]);
    subaxis(1,1,1,'ml',0.12, 'mb', 0.2, 'mt', 0.05, 'mr', 0.05);
    xdata = xmin:dx:xmax;
    p = plot(xdata,f_temp(xdata)); %plotting learned function
    set(p,'LineWidth',2, 'Color','blue')
    hold on
    % plotting taylors approximations over the learned function
    Dx = (xmax-xmin)/m;
    x_expansions = xmin+Dx/2:Dx:xmax-Dx/2;
    for part=1:1:m
        interval = xmin + (part-1) * Dx:dx*5:xmin + part * Dx;
        taylor_order = all_k_dict{m}(part);
        f_local = taylor(f_temp_symbolic, x, 'Order', 3,  'ExpansionPoint', x_expansions(part));
        f_local = matlabFunction(f_local);
        hold on
        p = plot(interval, f_local(interval),'-');
        grid on
        set(p,'LineWidth',2, 'Color','red', 'LineWidth', 1.5)
        set(gca,'XTick',linspace(xdata(1),xdata(end),11))
        % set(gca,'XTickLabel',round(linspace(tspan(1),tspan(end),11),2))
        xlabel('$x$', 'Interpreter','latex', 'fontsize', 10)
        ylabel('$y$', 'Interpreter','latex', 'fontsize', 10)
        leg1 = legend('$f_1$','$f_2$', '$f_3$', '$\tilde{f}_1$', '$\tilde{f}$', '$\tilde{f}_3$');
        set(leg1, 'Interpreter','latex');
        a=get(leg1,'position');
        set(leg1,'position',[0.8032, 0.5, 0.1211, 0.1509]);
        ylim(ylims);
        set(leg1,'Interpreter','latex');
        set(leg1,'FontSize', 10);
        filename = sprintf('Talor_%d_sections_3D.pdf',m);
        print(filename, '-dpdf');
        movefile(filename,'./figs') 
    end
end
