% stattest.m
clear;
datapath = "../";
addpath(datapath);

% List of .mat filenames
mat_files = {
    'final_Reg.mat',
    'final_NM.mat',
    'final_EM.mat'
};

num_files = numel(mat_files);

methods  = ["RKF","NM","EM"];

for idx = 1:num_files
    figure(idx);
    S = load(fullfile(datapath, mat_files{idx}));
    times = cellfun(@double, S.times);
    TT = length(times);
    plot3DCurve(times(1:TT),(1:size(S.fAll,1))/365 ,S.fAll(1:end,:), methods(idx));
end



