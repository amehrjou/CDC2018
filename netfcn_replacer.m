%{
Author: Arash Mehrjou
This file is intended to replace two lines of netfcn.m and make it usable
% for Taylor expansion.
%}
function netfcn_replacer(ddim)
% Input: The dimension of the learned dynamical system.
% Output: Two lines in the file netfcn.m s changed based on the
% input dimension. netfcn.m is the auto generated file by the trained
% neural network.
C = regexp( fileread('netfcn.m'), '\n', 'split');
hlayers_indices = ~cellfun(@isempty,regexp(C,'^a\d '));
num_hlayers = sum(hlayers_indices);
all_hlayers = C(hlayers_indices);
last_hlayer = all_hlayers(num_hlayers);
Ind1_to_replace = contains(C,'xp1 =');
Ind2_to_replace = contains(C,'y1 =');
if ddim == 1
    replace_line1 = 'xp1 = (x1 - x1_step1.xoffset) * x1_step1.gain + x1_step1.ymin;';
    replace_line2 = sprintf('y1 = (a%d - y1_step1.ymin) / y1_step1.gain + y1_step1.xoffset;', num_hlayers);
else
    replace_line1 = 'xp1 = diag(x1_step1.gain) * (x1 - x1_step1.xoffset)  + x1_step1.ymin;';
    replace_line2 = sprintf('y1 = diag(y1_step1.gain)^-1 * (a%d - y1_step1.ymin) + y1_step1.xoffset;', num_hlayers);
end
C{Ind1_to_replace} = replace_line1;
C{Ind2_to_replace} = replace_line2;
fileID = fopen('netfcn.m','w');
formatSpec = '%s\n';
[nrows,ncols] = size(C);
for row = 1:ncols
    fprintf(fileID,formatSpec,C{row});
end
fclose(fileID);
end