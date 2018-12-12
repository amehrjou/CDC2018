%{
Author: Arash Mehrjou
This function takes a local interval as input and computes the optimal
order of taylor expansion for that interval between min_k and max_k. This
functio is specifically written for Lorenz dynamical system. It returns
x_orig exceptionally compared to other choosek files for other dimensions.
%}

function [numOfSums, Lopt, xhat, xorig] = choosek_NN3D(tspan, f, f_orig, fvars, x0, timestep, lambda, min_k, max_k)
%L = L_complexity + L_state 
%L_complexity = k * lambda, where lambda is a weighting factor
%L_state = int_0^T || x(t) - xhat(t) || dt
    ftobetaylored = f;
%     x = sym('x', [1,8]);
    x = fvars;
    f = matlabFunction(f);
    f = @(t,x)f(x(1),x(2),x(3));
    [t,xtrue] = ode45(f, tspan, x0 );
    savek = inf;
    saveL = inf;
    for k=min_k:max_k
        L_state = 0;
        t10 = taylor(ftobetaylored, x, 'Order', k,  'ExpansionPoint', x0);
        fhat = matlabFunction(t10);
        fhat = @(t,x)fhat(x(1),x(2),x(3));
        [t,xhat] = ode45(fhat,tspan, x0 );
        for i = 1 : size(xtrue,1)-1
            %Riemann sum approximation for integral in L_state
            L_state = L_state + norm( xtrue(i,:) - xhat(i,:) ) * timestep;
        end
        L_complexity = k * lambda;
        L = L_complexity + L_state;
        if( L < saveL )
            saveL = L;
            savek = k;
        end
    end
    numOfSums = savek;
    Lopt = saveL;
    % creating xT
    t10 = taylor(ftobetaylored, x, 'Order', savek,  'ExpansionPoint', x0);
    fhat = matlabFunction(t10);
    fhat = @(t,x)fhat(x(1),x(2),x(3));
    [t,xhat] = ode45(fhat,tspan, x0);
    [t,xorig] = ode45(f_orig,tspan, x0);
    xT = xhat(end,:);
    
    
