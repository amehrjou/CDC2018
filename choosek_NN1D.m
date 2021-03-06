%{
Author: Arash Mehrjou
This function takes a local interval as input and computes the optimal
order of taylor expansion for that interval between min_k and max_k
%}

function [numOfSums, Lopt, xhat] = choosek_NN1D(tspan, f, fvars, x0, timestep, lambda, min_k, max_k)
%L = L_complexity + L_state 
%L_complexity = k * lambda, where lambda is a weighting factor
%L_state = int_0^T || x(t) - xhat(t) || dt

    ftobetaylored = f;
    x = fvars;
    f = matlabFunction(f);
    f = @(t,x)f(x(1));
    [t,xtrue] = ode45(f,tspan, x0 );
    savek = inf;
    saveL = inf;
    for k=min_k:max_k
        L_state = 0;
        t10 = taylor(ftobetaylored, x, 'Order', k,  'ExpansionPoint', x0);
        fhat = matlabFunction(t10);
        fhat = @(t,x)fhat(x(1));
        [t,xhat] = ode45(fhat,tspan, x0 ); 
        for i = 1 : size(xtrue,1)-1
            L_state = L_state + norm( xtrue(i,:) - xhat(i,:) ) * timestep; % Riemann sum approximation for integral in L_state
        end
        L_complexity = k * lambda;
        L = L_complexity + L_state;
%         fprintf('k=%d, L_comlexity:%7.7f | L_state:%7.7f\n', k, L_complexity, L_state)
        if( L < saveL )
            saveL = L;
            savek = k;
        end
    end
    numOfSums = savek;
    Lopt = saveL;
    %% Creating xT
    t10 = taylor(ftobetaylored, x, 'Order', savek,  'ExpansionPoint', x0);
    fhat = matlabFunction(t10);
    fhat = @(t,x)fhat(x(1));
    [t,xhat] = ode45(fhat,tspan, x0 );
    xT = xhat(end,:);
    
    
