% stattest.m
clear;
datapath = "../";
addpath(datapath);

% ——— PREALLOCATE combined containers ———
combined_innov.RKF = [];
combined_innov.NM  = [];
combined_innov.EM  = [];
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
    '2024-10-17_OOS_2025-04-09.mat'
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
    S = load(fullfile(datapath, mat_files{idx}));
    % convert times
    times = cellfun(@double, S.times);
    % pack innovations
    innovations.RKF = S.innovationAll_RKF;
    innovations.NM  = S.innovationAll_NM;
    innovations.EM  = S.innovationAll_EM;
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


%% ==================== Likelihood per‐file ====================
lik_models = ["NM","EM"];
lik_n      = numel(lik_models);

figure(2); clf;
for idx = 1:num_files
    if idx <= 8
        subplot_idx = idx;
    else
        % shift the last two to columns 2 and 3 in row 3
        subplot_idx = 8 + (idx - 8) + 1;  % 9 -> 10, 10 -> 11
    end
    S = load(fullfile(datapath, mat_files{idx}));
    % pack likelihoods
    likelihoods.NM = S.innovation_likelihood_NM;
    likelihoods.EM = S.innovation_likelihood_EM;
    % append to combined
    combined_lik.NM = [combined_lik.NM; likelihoods.NM(:)];
    combined_lik.EM = [combined_lik.EM; likelihoods.EM(:)];
    % compute α‐matrix
    [~,~,~, alpha_lik, sig_lik] = ...
      model_comparason_likelihood(lik_models, likelihoods, significance_level);
    % subplot
    subplot(rows, cols, subplot_idx);
    imagesc(alpha_lik);
    colormap('Abyss'); colorbar;
    xticks(1:lik_n); yticks(1:lik_n);
    xticklabels(lik_models); yticklabels(lik_models);
    title(strrep(mat_files{idx}, '_','\_'));
    % annotate
    for i=1:lik_n
      for j=1:lik_n
        text(j,i,sprintf('%.1f%%',alpha_lik(i,j) *100), ...
             'HorizontalAlignment','center','Color','white');
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
