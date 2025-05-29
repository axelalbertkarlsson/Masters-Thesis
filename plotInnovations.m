%% init + comb boxplot
mat_files = dir('*.mat');
nFiles = length(mat_files);

% Preallocate cell arrays to store innovations for each method
innovation_EM = {};
innovation_NM = {};
innovation_RKF = {};

for k = 1:nFiles
    % Load each file
    data = load(mat_files(k).name);
    
    % Store the innovationAll_* variables for each method
    if isfield(data, 'innovationAll_EM')
        innovation_EM{end+1} = data.innovationAll_EM;
    end
    if isfield(data, 'innovationAll_NM')
        innovation_NM{end+1} = data.innovationAll_NM;
    end
    if isfield(data, 'innovationAll_RKF')
        innovation_RKF{end+1} = data.innovationAll_RKF;
    end
end

all_innov_EM = [];
for file_k = 1:length(innovation_EM)
    innov_cell = innovation_EM{file_k};
    for t = 1:length(innov_cell)
        all_innov_EM = [all_innov_EM; innov_cell{t}(:)];
    end
end

% Repeat for NM and RKF
all_innov_NM = [];
for file_k = 1:length(innovation_NM)
    innov_cell = innovation_NM{file_k};
    for t = 1:length(innov_cell)
        all_innov_NM = [all_innov_NM; innov_cell{t}(:)];
    end
end

all_innov_RKF = [];
for file_k = 1:length(innovation_RKF)
    innov_cell = innovation_RKF{file_k};
    for t = 1:length(innov_cell)
        all_innov_RKF = [all_innov_RKF; innov_cell{t}(:)];
    end
end

innov_mat = [all_innov_EM, all_innov_NM, all_innov_RKF];

figure;
boxH = boxplot(innov_mat, 'Labels', {'EM', 'NM', 'RKF'}, 'Symbol', '+');

% Define colors
bright_blue = [52, 152, 219]/255;           % For the boxes
lighter_navy = [40, 80, 130]/255;           % For the outliers (lighter navy)

% Color all boxes bright blue
boxes = findobj(gca, 'Tag', 'Box');
for j = 1:length(boxes)
    patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), bright_blue, ...
        'FaceAlpha', 0.85, 'EdgeColor', bright_blue, 'LineWidth', 2);
end

% Set median lines to white for contrast (optional)
medians = findobj(gca, 'Tag', 'Median');
for j = 1:length(medians)
    set(medians(j), 'Color', [1 1 1], 'LineWidth', 2);
end

% Set all outlier markers to lighter navy
hOutliers = findobj(gca, 'Tag', 'Outliers');
for j = 1:length(hOutliers)
    set(hOutliers(j), 'MarkerEdgeColor', lighter_navy);
end

xlabel('Method');
ylabel('Innovation Value');
title('Comparison of Innovations for EM, NM, RKF');
grid on;

%% Histograms
nbins = 10000;
figure;
histogram(all_innov_EM, nbins, 'FaceColor', [52, 152, 219]/255);
title('EM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;

figure;
histogram(all_innov_NM, nbins, 'FaceColor', [46, 204, 113]/255);
title('NM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;

figure;
histogram(all_innov_RKF, nbins, 'FaceColor', [155, 89, 182]/255);
title('RKF Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;

%% Histograms all together in subplots
nbins = 10000;
figure;

subplot(1,3,1);
histogram(all_innov_EM, nbins, 'FaceColor', [52, 152, 219]/255);
title('EM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.003, 0.003]);

subplot(1,3,2);
histogram(all_innov_NM, nbins, 'FaceColor', [40, 80, 130]/255);  
title('NM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.003, 0.003]);

subplot(1,3,3);
histogram(all_innov_RKF, nbins, 'FaceColor', [155, 89, 182]/255);
title('RKF Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.003, 0.003]);

sgtitle('Histogram of Innovations');
%% Histograms all together zoomed
nbins = 300000;
figure;
hold on;
histogram(all_innov_EM, nbins, 'FaceColor', [12, 30, 51]/255, 'FaceAlpha', 1.0, 'EdgeColor', 'none');
histogram(all_innov_NM, nbins, 'FaceColor', [37, 139, 219]/255, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
histogram(all_innov_RKF, nbins, 'FaceColor', [40, 80, 130]/255, 'FaceAlpha', 0.7, 'EdgeColor', 'none');
hold off;

title('Histogram of Innovations (Zoomed In)');
xlabel('Innovation');
ylabel('Count');
xlim([-0.0005, 0.0005]);
grid on;

% Custom colorbox legend
x0 = 0.78; y0 = 0.75; dy = 0.05;
em_color  = [12, 30, 51]/255;
nm_color  = [37, 139, 219]/255;
rkf_color = [40, 80, 130]/255;

annotation('rectangle', [x0, y0, 0.025, 0.035], 'FaceColor', em_color,  'EdgeColor', 'none');
annotation('rectangle', [x0, y0-dy, 0.025, 0.035], 'FaceColor', nm_color,  'EdgeColor', 'none');
annotation('rectangle', [x0, y0-2*dy, 0.025, 0.035], 'FaceColor', rkf_color, 'EdgeColor', 'none');
annotation('textbox', [x0+0.03, y0, 0.05, 0.035], 'String', 'EM',  'LineStyle', 'none', 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
annotation('textbox', [x0+0.03, y0-dy, 0.05, 0.035], 'String', 'NM',  'LineStyle', 'none', 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
annotation('textbox', [x0+0.03, y0-2*dy, 0.05, 0.035], 'String', 'RKF', 'LineStyle', 'none', 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');


%% Histograms zoomed
figure;
histogram(all_innov_EM, nbins, 'FaceColor', [52, 152, 219]/255);
title('EM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.0005, 0.0005]);

figure;
histogram(all_innov_NM, nbins, 'FaceColor', [46, 204, 113]/255);
title('NM Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.0005, 0.0005]);

figure;
histogram(all_innov_RKF, nbins, 'FaceColor', [155, 89, 182]/255);
title('RKF Innovations');
xlabel('Innovation');
ylabel('Count');
grid on;
xlim([-0.0005, 0.0005]);

%% QQ plots
figure;
qqplot(all_innov_EM);
title('QQ Plot: EM Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

figure;
qqplot(all_innov_NM);
title('QQ Plot: NM Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

figure;
qqplot(all_innov_RKF);
title('QQ Plot: RKF Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

%% QQ plots all combined as subplots
figure;

subplot(1,3,1);
qqplot(all_innov_EM);
title('QQ Plot: EM Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

subplot(1,3,2);
qqplot(all_innov_NM);
title('QQ Plot: NM Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

subplot(1,3,3);
qqplot(all_innov_RKF);
title('QQ Plot: RKF Innovations');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;

sgtitle('QQ Plots of Innovations (EM, NM, RKF)');

%% QQ plot against student_t
nu = 2;  % degrees of freedom, change as needed

data_sets = {all_innov_EM, all_innov_NM, all_innov_RKF};
titles = {'EM Innovations', 'NM Innovations', 'RKF Innovations'};

figure;
for i = 1:3
    subplot(1,3,i);
    data = sort(data_sets{i});
    n = length(data);
    % Theoretical quantiles for Student's t
    p = ((1:n) - 0.5) / n;
    theor = tinv(p, nu);

    plot(theor, data, '+', 'Color', [0 0.447 0.741], 'MarkerSize', 6, 'LineWidth', 1); hold on;
    % Add a least-squares fit line (optional, robust)
    coeffs = polyfit(theor, data, 1);
    fitline = polyval(coeffs, theor);
    plot(theor, fitline, '--', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
    hold off;
    xlabel('Theoretical Quantiles (Student-t)');
    ylabel('Sample Quantiles');
    title(['QQ Plot: ' titles{i} ', \nu = ' num2str(nu)]);
    grid on;
end

sgtitle('QQ Plots of Innovations vs Student-t Distribution');

%% Boxplot for each contract
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_innov_EM = cell(1, nContracts);
all_innov_NM = cell(1, nContracts);
all_innov_RKF = cell(1, nContracts);

for c = 1:nContracts
    temp_EM  = [];
    temp_NM  = [];
    temp_RKF = [];
    for p = 1:nPeriods
        innovs_EM  = innovation_EM{p};   % 258x1 cell, each cell 28x1 double
        innovs_NM  = innovation_NM{p};
        innovs_RKF = innovation_RKF{p};
        for t = 1:numel(innovs_EM)
            % Only use if all contracts are present for this day
            if length(innovs_EM{t}) == nContracts && ...
               length(innovs_NM{t}) == nContracts && ...
               length(innovs_RKF{t}) == nContracts
                temp_EM  = [temp_EM;  innovs_EM{t}(c)];
                temp_NM  = [temp_NM;  innovs_NM{t}(c)];
                temp_RKF = [temp_RKF; innovs_RKF{t}(c)];
            end
        end
    end
    all_innov_EM{c}  = temp_EM;
    all_innov_NM{c}  = temp_NM;
    all_innov_RKF{c} = temp_RKF;
end

% Define blue color
blue_col = [52, 152, 219]/255;           % For the boxes
lighter_navy = [40, 80, 130]/255;        % For the outliers (lighter navy)

for c = 1:nContracts
    data = [all_innov_EM{c}, all_innov_NM{c}, all_innov_RKF{c}];
    figure;
    boxplot(data, 'Labels', {'EM', 'NM', 'RKF'});
    title(['Innovations Boxplot for Contract ', num2str(c)]);
    xlabel('Method');
    ylabel('Innovation');
    grid on;

    % Only color the edges of the boxes blue, keep face transparent
    boxes = findobj(gca, 'Tag', 'Box');
    for j = 1:length(boxes)
        p = patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), blue_col, ...
            'EdgeColor', blue_col, 'LineWidth', 2);
        set(p, 'FaceColor', 'none');
    end

    % Color the outliers blue
    hOutliers = findobj(gca, 'Tag', 'Outliers');
    for j = 1:length(hOutliers)
        set(hOutliers(j), 'MarkerEdgeColor', lighter_navy);
    end

    % --- Manually draw black median lines for each group ---
    % Get x positions for medians based on boxplot layout
    % Find the data used for plotting (for accurate median calculation)
    for groupIdx = 1:size(data,2)
        med = median(data(:,groupIdx),'omitnan');
        % X-range for median bar (depends on how wide the box is)
        % For standard boxplots, x-coordinates are: groupIdx +/- 0.25
        line([groupIdx-0.25, groupIdx+0.25], [med med], 'Color', 'k', 'LineWidth', 1);
    end
end
%% 8 Subplots for different contract lenghts
nContracts = 28;
nPeriods = 10;

% Prepare arrays to hold data for each contract and method
all_innov_EM = cell(1, nContracts);
all_innov_NM = cell(1, nContracts);
all_innov_RKF = cell(1, nContracts);

for c = 1:nContracts
    temp_EM  = [];
    temp_NM  = [];
    temp_RKF = [];
    for p = 1:nPeriods
        innovs_EM  = innovation_EM{p};   % 258x1 cell, each cell 28x1 double
        innovs_NM  = innovation_NM{p};
        innovs_RKF = innovation_RKF{p};
        for t = 1:numel(innovs_EM)
            % Only use if all contracts are present for this day
            if length(innovs_EM{t}) == nContracts && ...
               length(innovs_NM{t}) == nContracts && ...
               length(innovs_RKF{t}) == nContracts
                temp_EM  = [temp_EM;  innovs_EM{t}(c)];
                temp_NM  = [temp_NM;  innovs_NM{t}(c)];
                temp_RKF = [temp_RKF; innovs_RKF{t}(c)];
            end
        end
    end
    all_innov_EM{c}  = temp_EM;
    all_innov_NM{c}  = temp_NM;
    all_innov_RKF{c} = temp_RKF;
end

% Define blue color
blue_col = [52, 152, 219]/255;           % For the boxes
lighter_navy = [40, 80, 130]/255;        % For the outliers (lighter navy)

% List of contracts to plot
selected_contracts = [1, 4, 8, 12, 16, 20, 24, 28];

figure;
for idx = 1:length(selected_contracts)
    c = selected_contracts(idx);
    data = [all_innov_EM{c}, all_innov_NM{c}, all_innov_RKF{c}];
    subplot(2,4,idx);
    boxplot(data, 'Labels', {'EM', 'NM', 'RKF'});
    title(['Contract ', num2str(c)]);
    xlabel('Method');
    ylabel('Innovation');
    grid on;

    % Only color the edges of the boxes blue, keep face transparent
    boxes = findobj(gca, 'Tag', 'Box');
    for j = 1:length(boxes)
        p = patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), blue_col, ...
            'EdgeColor', blue_col, 'LineWidth', 2);
        set(p, 'FaceColor', 'none');
    end

    % Color the outliers navy
    hOutliers = findobj(gca, 'Tag', 'Outliers');
    for j = 1:length(hOutliers)
        set(hOutliers(j), 'MarkerEdgeColor', lighter_navy);
    end

    % Manually draw black median lines for each group
    for groupIdx = 1:size(data,2)
        med = median(data(:,groupIdx),'omitnan');
        line([groupIdx-0.25, groupIdx+0.25], [med med], 'Color', 'k', 'LineWidth', 1);
    end
end
% Add a floating annotation title above all subplots
annotation('textbox', [0 0.93 1 0.07], ...
    'String', 'Comparison of Innovations for EM, NM, RKF for Selected Contracts', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'EdgeColor', 'none', ...
    'FitBoxToText', 'on');


%% 10 Box plots
% Custom colors
bright_blue = [52, 152, 219]/255;
lighter_navy = [40, 80, 130]/255;

f = figure;
for i = 1:10
    % Prepare data for this subplot
    tmp_EM  = innovation_EM{i};   % 258x1 cell, each 28x1 double
    tmp_NM  = innovation_NM{i};
    tmp_RKF = innovation_RKF{i};
    vec_EM  = cell2mat(tmp_EM);   % 28*258 x 1 double
    vec_NM  = cell2mat(tmp_NM);
    vec_RKF = cell2mat(tmp_RKF);
    innov_mat_num = [vec_EM, vec_NM, vec_RKF];

    % Make a subplot for each set
    subplot(2,5,i);
    boxplot(innov_mat_num, 'Labels', {'EM', 'NM', 'RKF'}, 'Symbol', '+');
    title(['Set ', num2str(i)]);
    xlabel('');
    ylabel('');

    % Custom coloring: blue edges, no fill
    boxes = findobj(gca, 'Tag', 'Box');
    for j = 1:length(boxes)
        p = patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), bright_blue, ...
            'FaceAlpha', 0.85, 'EdgeColor', bright_blue, 'LineWidth', 2);
        set(p, 'FaceColor', 'none');
    end

    % Draw black median lines for each group (EM, NM, RKF)
    for groupIdx = 1:size(innov_mat_num,2)
        med = median(innov_mat_num(:,groupIdx), 'omitnan');
        line([groupIdx-0.25, groupIdx+0.25], [med med], 'Color', 'k', 'LineWidth', 1);
    end

    % Color the outliers navy
    hOutliers = findobj(gca, 'Tag', 'Outliers');
    for j = 1:length(hOutliers)
        set(hOutliers(j), 'MarkerEdgeColor', lighter_navy);
    end
    grid on;
end

% Add a floating annotation title above all subplots
annotation('textbox', [0 0.93 1 0.07], ...
    'String', 'Comparison of Innovations for EM, NM, RKF (All Sets)', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'EdgeColor', 'none', ...
    'FitBoxToText', 'on');