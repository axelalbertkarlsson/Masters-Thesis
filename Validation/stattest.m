clear;

datapath = "../";

addpath(datapath);

% load 2006-08-16_OOS_2007-08-21
% load 2008-09-02_OOS_2009-09-04
% load 2010-09-08_OOS_2011-09-06.mat
% load 2012-09-07_OOS_2013-09-12.mat
% load 2014-09-18_OOS_2015-09-23.mat
% load 2016-09-26_OOS_2017-09-26.mat
% load 2018-10-02_OOS_2019-10-07.mat
% load 2020-10-12_OOS_2021-10-12.mat
% load 2022-10-12_OOS_2023-10-12.mat
% load 2024-10-17_OOS_2025-04-09.mat


significance_level = 0.05;
contract_index = 1; %Choose which of the 27 OIS or EONIA

model_list = ["RKF", "NM", "EM"];

innovations = struct();
innovations.RKF = innovationAll_RKF;
innovations.NM = innovationAll_NM;
innovations.EM = innovationAll_EM;

[bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = model_comparason(model_list, innovations, significance_level, contract_index);

% Create tables for easy viewing, replace later!!!
results_table_bar_d = array2table(bar_d_matrix, 'RowNames', cellstr(model_list), 'VariableNames', cellstr(model_list));
results_table_z = array2table(z_matrix, 'RowNames', cellstr(model_list), 'VariableNames', cellstr(model_list));
results_table_p = array2table(p_matrix, 'RowNames', cellstr(model_list), 'VariableNames', cellstr(model_list));
results_table_alpha = array2table(alpha_matrix, 'RowNames', cellstr(model_list), 'VariableNames', cellstr(model_list));

% Plot the confusion matrix as a heatmap
figure; % Use a different figure for the confusion matrix
clf; % Clear any existing plots in this figure
imagesc(alpha_matrix);
colormap('Winter'); % Choose a color map
colorbar; % Add a color bar

% Set axis labels
xticks(1:length(model_list));
yticks(1:length(model_list));
xticklabels(model_list);
yticklabels(model_list);
xlabel('Models');
ylabel('Models');
title('Probability of Outperformance with Significance Level = ', num2str(significance_level));

% Annotate the cells with the values
for i = 1:size(alpha_matrix, 1)
    for j = 1:size(alpha_matrix, 2)
        % Annotate the cell with the value
        text(j, i, sprintf('%.2f', alpha_matrix(i, j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', 'white', 'FontSize', 10);

        % Check significance and draw a red border if necessary
        if significance_matrix(i, j) == 0
            rectangle('Position', [j-0.5, i-0.5, 1, 1], ... % [x, y, width, height]
                      'EdgeColor', 'red', ...
                      'LineWidth', 2);
        end
    end
end
