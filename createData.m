function [xtrain ytrain] = createData(F, G, NPERIODS, MTrajecs, timestep)
%This function takes a SDE as input, s.t. dX_t = F(x_t) dt = Q dW_t and
%creates noisy data ( X(t_k), F(X(t_k) ) with k = 1,...,NPERIODS and does
%this for MTrajecs trajectories, which are started at randomized initial
%starting points.
    obj = sde(F, G); 
    paths = [];
    noiseIncrements = [];
    xtrain = [];
    ytrain = [];
    for i = 1 : MTrajecs
        obj.StartState = normrnd(0,3)
        [pathTemp, Times, noiseIncrementsTemp] = simByEuler(obj, NPERIODS, 'DeltaTime', 0.01);
        paths = [paths ; pathTemp];
        noiseIncrements = [noiseIncrements ; noiseIncrementsTemp];
        ytrainTemp = pathTemp(2:end) - pathTemp(1:end-1);
        xtrainTemp = pathTemp(1:end-1);
        xtrain = [xtrain ; xtrainTemp ];
        ytrain = [ytrain ; ytrainTemp ];
    end
ytrain = ytrain/ 0.01;
end
