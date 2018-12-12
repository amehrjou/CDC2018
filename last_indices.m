%{
Author: Arash Mehrjou
This function computes the last indices of each interval once a duration is
divided into several sub-intervals.
%}
function [indspan, tspan] = last_indices(t0, t1, tstep, nparts)
total_steps = (t1 - t0) / tstep;
total_tspan = t0:tstep:t1;
steps_per_section = floor(total_steps / nparts);
inds = 1:1:max(size(total_tspan));
s0 = 1;
if nparts==1
    s1 = length(total_tspan);
else
    s1 = steps_per_section;
end
indspan = cell(nparts,1);
indspan{1} = inds(s0:s1);
tspan = cell(nparts,1);
tspan{1} = total_tspan(inds(s0:s1));
for k=2:1:nparts
    if k<nparts
        s0 = s1+1;
        s1 = s1 + steps_per_section;
    else
        s0 = s1+1;
        s1 = length(inds);
    end
    indspan{k} = inds(s0:s1);
    tspan{k} = total_tspan(indspan{k});
end


