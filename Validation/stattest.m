%% stattest.m
clear;
datapath = "../";
addpath(datapath);

% ——— PREALLOCATE combined containers ———
combined_innov.RKF = [];
combined_innov.NM  = [];
combined_innov.EM  = [];
combined_lik.RKF    = [];
combined_lik.NM    = [];
combined_lik.EM    = [];

% List of .mat filenames
mat_files = {
    '2006-08-16_OOS_2007-08-21.mat',
    '2008-09-02_OOS_2009-09-04.mat',
    '2010-09-08_OOS_2011-09-06.mat',
    '2012-09-07_OOS_2013-09-12.mat',
    '2014-09-18_OOS_2015-09-23.mat',
    '2016-09-26_OOS_2017-09-26.mat',
    '2018-10-02_OOS_2019-10-07.mat',
    '2020-10-12_OOS_2021-10-12.mat',
    '2022-10-12_OOS_2023-10-12.mat',
    '2024-10-17_OOS_2025-04-17.mat'
};

num_files = numel(mat_files);
cols      = ceil(sqrt(num_files));
rows      = ceil(num_files/cols);

%% ======================= MSE per‐file =======================
significance_level = 0.05;
contract_index     = 14;
MSE_models         = ["RKF","NM","EM"];
MSE_n             = numel(MSE_models);

figure(1); clf;

for idx = 1:num_files
    if idx <= 8
        subplot_idx = idx;
    else
        % shift the last two to columns 2 and 3 in row 3
        subplot_idx = 8 + (idx - 8) + 1;  % 9 -> 10, 10 -> 11
    end
    % Load RKF and NM from regular data
    S_main = load(fullfile(datapath, mat_files{idx}));
    % Load EM from old_data
    S_em   = load(fullfile(datapath, 'old_data', mat_files{idx}));
    
    % convert times
    times = cellfun(@double, S_main.times);
    % pack innovations
    innovations.RKF = S_main.innovationAll_RKF;
    innovations.NM  = S_main.innovationAll_NM;
    innovations.EM  = S_em.innovationAll_EM;
    % append to combined
    combined_innov.RKF = [combined_innov.RKF; innovations.RKF(:)];
    combined_innov.NM  = [combined_innov.NM;  innovations.NM(:)];
    combined_innov.EM  = [combined_innov.EM;  innovations.EM(:)];
    % compute α‐matrix
    [~,~,~,~,~, alpha_MSE, sig_MSE] = ...
      model_comparason_MSE(MSE_models, innovations, significance_level, contract_index);
    % subplot
    subplot(rows, cols, subplot_idx);
    imagesc(alpha_MSE);
    colormap('Abyss'); colorbar;
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(MSE_models); yticklabels(MSE_models);
    title(strrep(mat_files{idx}, '_','\_'));
    % title(sprintf('Set: %d', idx));
    % annotate
    for i=1:MSE_n
      for j=1:MSE_n
        text(j,i,sprintf('%.1f%%',alpha_MSE(i,j) *100), ...
             'HorizontalAlignment','center','Color','white');
        if sig_MSE(i,j)==0
          rectangle('Position',[j-0.5,i-0.5,1,1], ...
                    'EdgeColor','red','LineWidth',2);
        end
      end
    end
end
sgtitle(sprintf('MSE Comparison (α=%.2f)', significance_level));
%% AXEL VERSION OF ABOVE
rows = 5;
cols = 2;
figure(1); clf;
S_test = load(fullfile(datapath, 'old_data', mat_files));
for idx = 1:num_files
    subplot(rows, cols, idx);  % simple linear indexing
    % Load RKF and NM from regular data
    S_main = load(fullfile(datapath, mat_files{idx}));
    % Load EM from old_data
    S_em   = load(fullfile(datapath, 'old_data', mat_files{idx}));
    
    % convert times
    times = cellfun(@double, S_main.times);
    % pack innovations
    innovations.RKF = S_main.innovationAll_RKF;
    innovations.NM  = S_main.innovationAll_NM;
    innovations.EM  = S_em.innovationAll_EM;
    % append to combined
    combined_innov.RKF = [combined_innov.RKF; innovations.RKF(:)];
    combined_innov.NM  = [combined_innov.NM;  innovations.NM(:)];
    combined_innov.EM  = [combined_innov.EM;  innovations.EM(:)];
    % compute α‐matrix
    [~,~,~,~,~, alpha_MSE, sig_MSE] = ...
      model_comparason_MSE(MSE_models, innovations, significance_level, contract_index);
    % subplot
    imagesc(alpha_MSE);
    colormap('Abyss'); colorbar;
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(MSE_models); yticklabels(MSE_models);
    % title(strrep(mat_files{idx}, '_','\_'));
    title(sprintf('Set: %d', idx));
    % annotate
    for i=1:MSE_n
      for j=1:MSE_n
        text(j,i,sprintf('%.1f%%',alpha_MSE(i,j) *100), ...
             'HorizontalAlignment','center','Color','white');
        if sig_MSE(i,j)==0
          rectangle('Position',[j-0.5,i-0.5,1,1], ...
                    'EdgeColor','red','LineWidth',2);
        end
      end
    end
end
sgtitle(sprintf('MSE Comparison (α=%.2f)', significance_level));
%% AXEL MSE but for each contract (Only data indjestion)
% ===== EM ==== %
datapath = "../";
olddatapath = fullfile(datapath, 'old_data');
% Get all .mat files in that directory
mat_files = dir(fullfile(olddatapath, '*.mat'));
% Preallocate cell array to hold data
S_Em = cell(1, numel(mat_files));
% Loop over and load each file
for k = 1:numel(mat_files)
    filename = fullfile(olddatapath, mat_files(k).name);
    S_Em{k} = load(filename);
end

% ==== REG ==== %
mat_files_REG = dir(fullfile(datapath, '*.mat'));
% Preallocate cell array to hold data
S_main = cell(1, numel(mat_files_REG));
% Loop over and load each file
for k = 1:numel(mat_files_REG)
    filename = fullfile(olddatapath, mat_files_REG(k).name);
    S_main{k} = load(filename);
end

% === Start of actual program === % 
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_innov_EM = cell(1, nContracts);
all_innov_NM = cell(1, nContracts);
all_innov_RKF = cell(1, nContracts);
% all_innovations = [all_innov_EM];

for c = 1:nContracts
    temp_EM  = [];
    temp_NM  = [];
    temp_RKF = [];
    for p = 1:nPeriods
        innovs_EM  = S_Em{1,p}.innovationAll_EM;
        innovs_NM  = S_main{1,p}.innovationAll_NM;
        innovs_RKF  = S_main{1,p}.innovationAll_RKF;
        for t = 1:numel(innovs_EM)
            % Only use if all contracts are present for this day
            if length(innovs_EM{t,1}) == nContracts
                temp_EM  = [temp_EM;  innovs_EM{t,1}(c)];
                temp_NM  = [temp_NM;  innovs_NM{t,1}(c)];
                temp_RKF  = [temp_RKF;  innovs_RKF{t,1}(c)];
            end
        end
    end
    all_innov_EM{c}  = temp_EM;
    all_innov_NM{c}  = temp_NM;
    all_innov_RKF{c}  = temp_RKF;
    all_innovations{c} = [temp_RKF, temp_NM, temp_EM];
end
%% Plot of above (28 different plots)
model_list = ["RKF","NM","EM"];
MSE_n = numel(model_list);
significance_level = 0.05;

for index = 1:nContracts
    data_for_c = all_innovations{index};  % N×3 matrix
    [~,~,~,~,~, alpha_MSE, sig_MSE] = ...
        model_comp_MSE_per_contract(model_list, data_for_c, significance_level);

    % Open a new figure for each contract
    figure(index); clf;
    imagesc(alpha_MSE);
    colormap('Abyss'); colorbar;
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(model_list); yticklabels(model_list);
    title(sprintf('Contract: %d', index));
    % Annotate
    for i = 1:MSE_n
        for j = 1:MSE_n
            text(j, i, sprintf('%.1f%%', alpha_MSE(i, j) * 100), ...
                 'HorizontalAlignment', 'center', 'Color', 'white');
            if sig_MSE(i, j) == 0
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                          'EdgeColor', 'red', 'LineWidth', 2);
            end
        end
    end
end
%% Plot of MSE but with the 8 contracts of the box plot
model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;
selected        = [1, 4, 8, 12, 16, 20, 24, 28];  % the contracts you care about

% Create a new figure
figure; clf;

% 2 rows × 4 columns
rows = 2; 
cols = 4;

for k = 1:numel(selected)
    idx = selected(k);
    
    % compute the MSE α‐matrix for this contract
    data_c = all_innovations{1,idx};  % N×3 matrix for contract idx
    [bar_d1,bar_d2,~,~,~, alpha_MSE, sig_MSE] = ...
        model_comp_MSE_per_contract(model_list, data_c, significance_lv, idx);
    
    meas_improve = MSE_measurable_improvement( ...
    model_list, bar_d1, bar_d2, alpha_MSE);
    % subplot in the k-th slot
    subplot(rows, cols, k);
    imagesc(alpha_MSE);
    colormap('Abyss');
    colorbar;

    % label axes
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(model_list);
    yticklabels(model_list);

    % title with the actual contract index
    title(sprintf('Contract %d', idx));

    % annotate percentages and significance boxes
    for i = 1:MSE_n
        for j = 1:MSE_n
            text(j, i, sprintf('%.1f%%', alpha_MSE(i,j)*100), ...
                 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
            if ~sig_MSE(i,j)
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                          'EdgeColor', 'red', 'LineWidth', 2);
            end
        end
    end
end

% Optional super‐title for the whole figure
sgtitle(sprintf('Pairwise MSE Statistical Test for Selected Contracts'));

%% Separate Table "plot" of Measurable Improvement for the 8 selected contracts
model_list      = ["RKF","NM","EM"];
significance_lv = 0.05;
selected        = [1, 4, 8, 12, 16, 20, 24, 28];

% Preallocate containers
meas_improve_all = cell(numel(selected),1);
alpha_all        = cell(numel(selected),1);

for k = 1:numel(selected)
    idx = selected(k);
    
    % pull in the innovations for this contract
    data_c = all_innovations{1,idx};  % N×3 matrix
    
    % compute MSE statistics
    [bar_d1, bar_d2, ~, ~, ~, alpha_MSE, ~] = ...
      model_comp_MSE_per_contract(model_list, data_c, significance_lv, idx);
    
    % compute measurable improvement
    meas_improve = MSE_measurable_improvement(model_list, bar_d1, bar_d2, alpha_MSE);
    
    % store
    meas_improve_all{k} = meas_improve;
    alpha_all{k}        = alpha_MSE;
end

% save to disk for later table‐making
save('MSE_results.mat', ...
     'meas_improve_all', 'alpha_all', 'selected', 'model_list');
%% Test for above

% 2) Create the figure and force the painters renderer so text stays as text:
figure('Renderer','painters'); clf;

model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;
selected        = [1, 4, 8, 12, 16, 20, 24, 28];  % the contracts you care about

rows = 2; 
cols = 4;

for k = 1:numel(selected)
    idx = selected(k);
    
    % compute the MSE α‐matrix for this contract
    data_c = all_innovations{1,idx};  % N×3 matrix for contract idx
    [~,~,~,~,~, alpha_MSE, sig_MSE] = ...
        model_comp_MSE_per_contract(model_list, data_c, significance_lv, idx);

    % subplot in the k-th slot
    subplot(rows, cols, k);
    imagesc(alpha_MSE);
    colormap('Abyss');
    colorbar;

    % label axes
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(model_list); yticklabels(model_list);

    % title with the actual contract index
    title(sprintf('Contract %d', idx));

    % annotate percentages and significance boxes
    for i = 1:MSE_n
        for j = 1:MSE_n
            text(j, i, sprintf('%.1f%%', alpha_MSE(i,j)*100), ...
                 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
            if ~sig_MSE(i,j)
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                          'EdgeColor', 'red', 'LineWidth', 2);
            end
        end
    end
end

% 3) Add a super‐title and then export as a vector SVG so the font is preserved:
sgtitle('Pairwise MSE Statistical Test for Selected Contracts');
set(gcf,'Renderer','painters');              % double-check renderer
print(gcf, 'MSE_summary.svg', '-dsvg'); 

%% ==================== LOAD DATA - MSE PER TIME PERIOD AXEL WAY ====================
% ===== EM ==== %
datapath = "../";
olddatapath = fullfile(datapath, 'old_data');
% Get all .mat files in that directory
mat_files = dir(fullfile(olddatapath, '*.mat'));
% Preallocate cell array to hold data
S_Em = cell(1, numel(mat_files));
% Loop over and load each file
for k = 1:numel(mat_files)
    filename = fullfile(olddatapath, mat_files(k).name);
    S_Em{k} = load(filename);
end

% ==== REG ==== %
mat_files_REG = dir(fullfile(datapath, '*.mat'));
% Preallocate cell array to hold data
S_main = cell(1, numel(mat_files_REG));
% Loop over and load each file
for k = 1:numel(mat_files_REG)
    filename = fullfile(olddatapath, mat_files_REG(k).name);
    S_main{k} = load(filename);
end

% === Start of actual program === % 
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_innov_EM_set = cell(1, nPeriods);
all_innov_NM_set = cell(1, nPeriods);
all_innov_RKF_set = cell(1, nPeriods);
% all_innovations = [all_innov_EM];

for c = 1:nPeriods
    temp_EM_set  = [];
    temp_NM_set  = [];
    temp_RKF_set = [];
    for p = 1:nContracts
        innovs_EM_set  = S_Em{1,c}.innovationAll_EM;
        innovs_NM_set  = S_main{1,c}.innovationAll_NM;
        innovs_RKF_set  = S_main{1,c}.innovationAll_RKF;
        for t = 1:numel(innovs_EM_set)
            % Only use if all contracts are present for this day
            % if length(innovs_EM_set{t,1}) == nContracts
                temp_EM_set  = [temp_EM_set;  innovs_EM_set{t,1}(c)];
                temp_NM_set  = [temp_NM_set;  innovs_NM_set{t,1}(c)];
                temp_RKF_set  = [temp_RKF_set;  innovs_RKF_set{t,1}(c)];
            % end
        end
    end
    all_innov_EM_set{c}  = temp_EM_set;
    all_innov_NM_set{c}  = temp_NM_set;
    all_innov_RKF_set{c}  = temp_RKF_set;
    all_innovations_set{c} = [temp_RKF_set, temp_NM_set, temp_EM_set];
end

%% ==================== Plot - MSE PER TIME PERIOD AXEL WAY ====================
%% Plot of MSE but with the 8 contracts of the box plot
model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;

nPeriods = numel(all_innovations_set); % Or set manually if not defined

% Choose subplot grid (adjust as needed)
rows = 5; 
cols = 2;

figure; clf;

for idx = 1:nPeriods
    data_c = all_innovations_set{1, idx}; % N×3 matrix for set idx
    [bar_d1,bar_d2,~,~,~, alpha_MSE, sig_MSE] = ...
        model_comparason_MSE(model_list, data_c, significance_lv, idx);

    subplot(rows, cols, idx);

    imagesc(alpha_MSE);
    colormap('Abyss');
    colorbar;

    % Label axes
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(model_list);
    yticklabels(model_list);

    % Title for each set
    title(sprintf('Set %d', idx));

    % Annotate percentages and significance boxes
    for i = 1:MSE_n
        for j = 1:MSE_n
            text(j, i, sprintf('%.1f%%', alpha_MSE(i,j)*100), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
            if ~sig_MSE(i,j)
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                    'EdgeColor', 'red', 'LineWidth', 2);
            end
        end
    end
end

sgtitle('Pairwise MSE Statistical Test for Each Set');

%% ==================== LOAD DATA - MSE COMBINED ====================
% ===== EM ==== %
datapath = "../";
olddatapath = fullfile(datapath, 'old_data');
% Get all .mat files in that directory
mat_files = dir(fullfile(olddatapath, '*.mat'));
% Preallocate cell array to hold data
S_Em = cell(1, numel(mat_files));
% Loop over and load each file
for k = 1:numel(mat_files)
    filename = fullfile(olddatapath, mat_files(k).name);
    S_Em{k} = load(filename);
end

% ==== REG ==== %
mat_files_REG = dir(fullfile(datapath, '*.mat'));
% Preallocate cell array to hold data
S_main = cell(1, numel(mat_files_REG));
% Loop over and load each file
for k = 1:numel(mat_files_REG)
    filename = fullfile(olddatapath, mat_files_REG(k).name);
    S_main{k} = load(filename);
end

% === Start of actual program === % 
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_innov_EM_set = cell(1, nPeriods);
all_innov_NM_set = cell(1, nPeriods);
all_innov_RKF_set = cell(1, nPeriods);
% all_innovations = [all_innov_EM];

for c = 1:nPeriods
    temp_EM_set  = [];
    temp_NM_set  = [];
    temp_RKF_set = [];
    for p = 1:nContracts
        innovs_EM_set  = S_Em{1,c}.innovationAll_EM;
        innovs_NM_set  = S_main{1,c}.innovationAll_NM;
        innovs_RKF_set  = S_main{1,c}.innovationAll_RKF;
        for t = 1:numel(innovs_EM_set)
            % Only use if all contracts are present for this day
            % if length(innovs_EM_set{t,1}) == nContracts
                temp_EM_set  = [temp_EM_set;  innovs_EM_set{t,1}(c)];
                temp_NM_set  = [temp_NM_set;  innovs_NM_set{t,1}(c)];
                temp_RKF_set  = [temp_RKF_set;  innovs_RKF_set{t,1}(c)];
            % end
        end
    end
    all_innov_EM_set{c}  = temp_EM_set;
    all_innov_NM_set{c}  = temp_NM_set;
    all_innov_RKF_set{c}  = temp_RKF_set;
    all_innovations_set{c} = [temp_RKF_set, temp_NM_set, temp_EM_set];
end

all_innovations_combined = [];
for idx = 1:numel(all_innovations_like)
    all_innovations_combined = [all_innovations_combined; all_innovations_set{idx}];
end

%% ==================== PLOT - MSE COMBINED ====================
model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;

[bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = ...
    model_comparason_MSE(model_list, all_innovations_combined, significance_lv, 0);

figure; clf;
imagesc(alpha_matrix);
colormap('Abyss');
colorbar;

xticks(1:MSE_n); yticks(1:MSE_n);
xticklabels(model_list);
yticklabels(model_list);

title('Combined Pairwise MSE Across All Periods and Contracts');

for i = 1:MSE_n
    for j = 1:MSE_n
        text(j, i, sprintf('%.1f%%', alpha_matrix(i,j)*100), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
        if ~isnan(significance_matrix(i,j)) && ~significance_matrix(i,j)
            rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                'EdgeColor', 'red', 'LineWidth', 2);
        end
    end
end

%% ==================== LOAD DATA - Likelihood per‐file ====================
% ==== MAIN ==== %
datapath = "../";
mat_files_REG = dir(fullfile(datapath, '*.mat'));
% Preallocate cell array to hold data
S_main = cell(1, numel(mat_files_REG));
% Loop over and load each file
for k = 1:numel(mat_files_REG)
    filename = fullfile(mat_files_REG(k).name);
    S_main{k} = load(filename);
end

% === Start of actual program === % 
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_likelihood_EM = cell(1, nContracts);
all_likelihood_NM = cell(1, nContracts);
all_likelihood_RKF = cell(1, nContracts);
% all_innovations = [all_innov_EM];

for c = 1:nPeriods
    temp_EM  = [];
    temp_NM  = [];
    temp_RKF = [];
    for p = 1:nContracts
        innovs_like_EM  = S_main{1,c}.innovation_likelihood_EM;
        innovs_like_NM  = S_main{1,c}.innovation_likelihood_NM;
        innovs_like_RKF  = S_main{1,c}.innovation_likelihood_RKF;
        for t = 1:numel(innovs_like_EM)
            temp_EM  = [temp_EM;  innovs_like_EM(t)];
            temp_NM  = [temp_NM;  innovs_like_NM(t)];
            temp_RKF  = [temp_RKF;  innovs_like_RKF(t)];
        end
    end
    all_innov_like_EM{c}  = temp_EM;
    all_innov_like_NM{c}  = temp_NM;
    all_innov_like_RKF{c}  = temp_RKF;
    all_innovations_like{c} = [temp_RKF, temp_NM, temp_EM];
end

%% ========== Likelihood per‐file PLOT ========== %%
model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;

nPeriods = numel(all_innovations_like);

% Choose subplot grid
rows = 5; 
cols = 2;

figure; clf;

for idx = 1:nPeriods
    data_c = all_innovations_like{1, idx}; % N×3 matrix for set idx
    
    % Call your likelihood comparison function
    [bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = ...
        model_comparason_likelihood(model_list, data_c, significance_lv);

    subplot(rows, cols, idx);

    % Plot alpha_matrix, just as you did for MSE
    imagesc(alpha_matrix);
    colormap('Abyss');
    colorbar;

    % Label axes
    xticks(1:MSE_n); yticks(1:MSE_n);
    xticklabels(model_list);
    yticklabels(model_list);

    % Title for each set
    title(sprintf('Set %d', idx));

    % Annotate percentages and significance boxes
    for i = 1:MSE_n
        for j = 1:MSE_n
            text(j, i, sprintf('%.1f%%', alpha_matrix(i,j)*100), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
            % if ~significance_matrix(i,j)
            if ~isnan(significance_matrix(i,j)) && ~significance_matrix(i,j)
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                    'EdgeColor', 'red', 'LineWidth', 2);
            end
        end
    end
end

sgtitle('Pairwise Likelihood Statistical Test for Each Set');
%% ==================== LOAD DATA - Likelihood combined ====================
% ==== MAIN ==== %
datapath = "../";
mat_files_REG = dir(fullfile(datapath, '*.mat'));
% Preallocate cell array to hold data
S_main = cell(1, numel(mat_files_REG));
% Loop over and load each file
for k = 1:numel(mat_files_REG)
    filename = fullfile(mat_files_REG(k).name);
    S_main{k} = load(filename);
end

% === Start of actual program === % 
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_likelihood_EM = cell(1, nContracts);
all_likelihood_NM = cell(1, nContracts);
all_likelihood_RKF = cell(1, nContracts);
% all_innovations = [all_innov_EM];

for c = 1:nPeriods
    temp_EM  = [];
    temp_NM  = [];
    temp_RKF = [];
    for p = 1:nContracts
        innovs_like_EM  = S_main{1,c}.innovation_likelihood_EM;
        innovs_like_NM  = S_main{1,c}.innovation_likelihood_NM;
        innovs_like_RKF  = S_main{1,c}.innovation_likelihood_RKF;
        for t = 1:numel(innovs_like_EM)
            temp_EM  = [temp_EM;  innovs_like_EM(t)];
            temp_NM  = [temp_NM;  innovs_like_NM(t)];
            temp_RKF  = [temp_RKF;  innovs_like_RKF(t)];
        end
    end
    all_innov_like_EM{c}  = temp_EM;
    all_innov_like_NM{c}  = temp_NM;
    all_innov_like_RKF{c}  = temp_RKF;
    all_innovations_like{c} = [temp_RKF, temp_NM, temp_EM];
end

all_data = [];
for idx = 1:numel(all_innovations_like)
    all_data = [all_data; all_innovations_like{idx}];
end

%% ==================== PLOT - Likelihood combined ====================
model_list      = ["RKF","NM","EM"];
MSE_n           = numel(model_list);
significance_lv = 0.05;

[bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = ...
    model_comparason_likelihood(model_list, all_data, significance_lv);

figure; clf;
imagesc(alpha_matrix);
colormap('Abyss');
colorbar;

xticks(1:MSE_n); yticks(1:MSE_n);
xticklabels(model_list);
yticklabels(model_list);

title('Combined Pairwise Likelihood Across All Periods and Contracts');

for i = 1:MSE_n
    for j = 1:MSE_n
        text(j, i, sprintf('%.1f%%', alpha_matrix(i,j)*100), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 13);
        if ~isnan(significance_matrix(i,j)) && ~significance_matrix(i,j)
            rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                'EdgeColor', 'red', 'LineWidth', 2);
        end
    end
end
%% ==================== Likelihood per‐file ====================
% lik_models = ["NM","EM"];
% lik_n      = numel(lik_models);
% 
% figure(2); clf;
% for idx = 1:num_files
%     if idx <= 8
%         subplot_idx = idx;
%     else
%         % shift the last two to columns 2 and 3 in row 3
%         subplot_idx = 8 + (idx - 8) + 1;  % 9 -> 10, 10 -> 11
%     end
%     S = load(fullfile(datapath, mat_files{idx}));
%     % pack likelihoods
%     likelihoods.NM = S.innovation_likelihood_NM;
%     likelihoods.EM = S.innovation_likelihood_EM;
%     % append to combined
%     combined_lik.NM = [combined_lik.NM; likelihoods.NM(:)];
%     combined_lik.EM = [combined_lik.EM; likelihoods.EM(:)];
%     % compute α‐matrix
%     [~,~,~, alpha_lik, sig_lik] = ...
%       model_comparason_likelihood(lik_models, likelihoods, significance_level);
%     % subplot
%     subplot(rows, cols, subplot_idx);
%     imagesc(alpha_lik);
%     colormap('Abyss'); colorbar;
%     xticks(1:lik_n); yticks(1:lik_n);
%     xticklabels(lik_models); yticklabels(lik_models);
%     title(strrep(mat_files{idx}, '_','\_'));
%     % annotate
%     for i=1:lik_n
%       for j=1:lik_n
%         text(j,i,sprintf('%.1f%%',alpha_lik(i,j) *100), ...
%              'HorizontalAlignment','center','Color','white','FontSize', 12);
%         if sig_lik(i,j)==0
%           rectangle('Position',[j-0.5,i-0.5,1,1], ...
%                     'EdgeColor','red','LineWidth',2);
%         end
%       end
%     end
% end
% sgtitle(sprintf('Log-Likelihood Comparison (α=%.2f)', significance_level));

%% AXEL VERSION OF ABOVE 
lik_models = ["RKF", "NM","EM"];
lik_n      = numel(lik_models);
rows = 5;
cols = 3;

figure(2); clf;
for idx = 1:num_files
    subplot(rows, cols, idx);
    S = load(fullfile(datapath, mat_files{idx})); % Changed to the new file only
    % pack likelihoods
    likelihoods.RKF = S.innovation_likelihood_RKF;
    likelihoods.NM = S.innovation_likelihood_NM;
    likelihoods.EM = S.innovation_likelihood_EM;
    % append to combined
    combined_lik.RKF = [combined_lik.RKF; likelihoods.RKF(:)];
    combined_lik.NM = [combined_lik.NM; likelihoods.NM(:)];
    combined_lik.EM = [combined_lik.EM; likelihoods.EM(:)];
    % compute α‐matrix
    [~,~,~, alpha_lik, sig_lik] = ...
      model_comparason_likelihood(lik_models, likelihoods, significance_level);
    % subplot
    imagesc(alpha_lik);
    colormap('Abyss'); colorbar;
    xticks(1:lik_n); yticks(1:lik_n);
    xticklabels(lik_models); yticklabels(lik_models);
    title(strrep(mat_files{idx}, '_','\_'));
    % annotate
    for i=1:lik_n
      for j=1:lik_n
        text(j,i,sprintf('%.1f%%',alpha_lik(i,j) *100), ...
             'HorizontalAlignment','center','Color','white', ...
             'FontSize', 12);
        if sig_lik(i,j)==0
          rectangle('Position',[j-0.5,i-0.5,1,1], ...
                    'EdgeColor','red','LineWidth',2);
        end
      end
    end
end
sgtitle(sprintf('Log-Likelihood Comparison (α=%.2f)', significance_level));

%% ========== COMBINED TESTS ON GLUED INNOVATIONS & LIKELIHOODS ==========
% Combined MSE
figure(3); clf;
[~,~,~,~,~, alpha_MSE_c, sig_MSE_c] = ...
  model_comparason_MSE(MSE_models, combined_innov, significance_level, contract_index);
imagesc(alpha_MSE_c);
colormap('Abyss'); colorbar;
xticks(1:MSE_n); yticks(1:MSE_n);
xticklabels(MSE_models); yticklabels(MSE_models);
sgtitle(sprintf('Combined: MSE Comparison (α=%.2f)', significance_level));
for i=1:MSE_n
  for j=1:MSE_n
    text(j,i,sprintf('%.1f%%',alpha_MSE_c(i,j) *100), ...
         'HorizontalAlignment','center','Color','white');
    if sig_MSE_c(i,j)==0
      rectangle('Position',[j-0.5,i-0.5,1,1], ...
                'EdgeColor','red','LineWidth',2);
    end
  end
end

% Combined Likelihood
figure(5); clf;
[~,~,~, alpha_lik_c, sig_lik_c] = ...
  model_comparason_likelihood(lik_models, combined_lik, significance_level);
imagesc(alpha_lik_c);
colormap('Abyss'); colorbar;
xticks(1:lik_n); yticks(1:lik_n);
xticklabels(lik_models); yticklabels(lik_models);
sgtitle(sprintf('Combined: Log-Likelihood Comparison (α=%.2f)', significance_level));
for i=1:lik_n
  for j=1:lik_n
    text(j,i,sprintf('%.1f%%',alpha_lik_c(i,j) *100), ...
         'HorizontalAlignment','center','Color','white');
    if sig_lik_c(i,j)==0
      rectangle('Position',[j-0.5,i-0.5,1,1], ...
                'EdgeColor','red','LineWidth',2);
    end
  end
end
