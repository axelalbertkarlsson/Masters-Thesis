clear;

datapath = "../";
addpath(datapath);

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
    '2024-10-17_OOS_2025-04-09.mat'
};

%% MSE

significance_level = 0.05;
contract_index = 14;
MSE_model_list = ["RKF", "NM", "EM"];
MSE_num_models = length(MSE_model_list);

% Figure setup
num_files = length(mat_files);
cols = ceil(sqrt(num_files));
rows = ceil(num_files / cols);
figure(1);
clf;

meas_improvements = cell(length(num_files));

for idx = 1:num_files
    % Load file
    file_to_load = fullfile(datapath, mat_files{idx});
    load(file_to_load);

    % Assign innovations
    innovations = struct();
    innovations.RKF = innovationAll_RKF;
    innovations.NM = innovationAll_NM;
    innovations.EM = innovationAll_EM;

    % Compute comparison matrices
    [bar_d_matrix_1, bar_d_matrix_2, ~, ~, ~, MSE_alpha_matrix, MSE_significance_matrix] = model_comparason_MSE(MSE_model_list, innovations, significance_level, contract_index);


    % Code for Measurable Improvement

    meas_improvements{idx} = MSE_measurable_improvement(MSE_model_list, bar_d_matrix_1, bar_d_matrix_2, MSE_alpha_matrix);

    % Plot in subplot
    subplot(rows, cols, idx);
    imagesc(MSE_alpha_matrix);
    colormap('Winter');
    colorbar;
    
    % Set axis
    xticks(1:MSE_num_models);
    yticks(1:MSE_num_models);
    xticklabels(MSE_model_list);
    yticklabels(MSE_model_list);
    title(strrep(mat_files{idx}, '_', '\_'));

    % Annotate cells
    for i = 1:MSE_num_models
        for j = 1:MSE_num_models
            text(j, i, sprintf('%.2f', MSE_alpha_matrix(i, j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color', 'white', 'FontSize', 10);
            if MSE_significance_matrix(i, j) == 0
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                          'EdgeColor', 'red', ...
                          'LineWidth', 2);
            end
        end
    end
end

sgtitle(sprintf('MSE Model Comparison Alpha Matrices (Contract = %d, Significance = %.2f)', contract_index, significance_level));

%% Likelihood

significance_level = 0.05;
likelihood_model_list = ["NM", "EM"];
likelihood_num_models = length(likelihood_model_list);

% Figure setup
num_files = length(mat_files);
cols = ceil(sqrt(num_files));
rows = ceil(num_files / cols);
figure(2);
clf;

for idx = 1:num_files
    % Load file
    file_to_load = fullfile(datapath, mat_files{idx});
    load(file_to_load);

    % Assign innovations
    likelihoods = struct();
    likelihoods.NM = innovation_likelihood_NM;
    likelihoods.EM = innovation_likelihood_EM;

    % Compute comparison matrices
    [~, ~, ~, likelihood_alpha_matrix, likelihood_significance_matrix] = model_comparason_likelihood(likelihood_model_list, likelihoods, significance_level);

    % Plot in subplot
    subplot(rows, cols, idx);
    imagesc(likelihood_alpha_matrix);
    colormap('Winter');
    colorbar;
    
    % Set axis
    xticks(1:likelihood_num_models);
    yticks(1:likelihood_num_models);
    xticklabels(likelihood_model_list);
    yticklabels(likelihood_model_list);
    title(strrep(mat_files{idx}, '_', '\_'));

    % Annotate cells
    for i = 1:likelihood_num_models
        for j = 1:likelihood_num_models
            text(j, i, sprintf('%.2f', likelihood_alpha_matrix(i, j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color', 'white', 'FontSize', 10);
            if likelihood_significance_matrix(i, j) == 0
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                          'EdgeColor', 'red', ...
                          'LineWidth', 2);
            end
        end
    end
end

sgtitle(sprintf('Log-Likelihood Model Comparison Alpha Matrices (Significance = %.2f)', significance_level));
