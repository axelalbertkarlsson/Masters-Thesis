function [bar_d_matrix_1, bar_d_matrix_2,bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = model_comparason_MSE(model_list, innovations, significance_level, contract_index)
    % Number of models
    T = length(innovations.(model_list(1)));

    model_count = length(model_list);

    % Initialize results matrices
    bar_d_matrix_1 = NaN(model_count, model_count);
    bar_d_matrix_2 = NaN(model_count, model_count);
    z_matrix     = NaN(model_count, model_count);
    p_matrix     = NaN(model_count, model_count);
    alpha_matrix = NaN(model_count, model_count);
    significance_matrix = NaN(model_count, model_count);
 
    % Loop through model pairs
    for i = 1:model_count
        for j = 1:model_count
            if i == j
                continue; % Skip same-model comparison
            end

            model_1 = model_list(i);
            model_2 = model_list(j);

            epsilon_i = innovations.(model_1);
            epsilon_j= innovations.(model_2);

            % Compute d_{i,k} via double loop
            d = zeros(T);
            d_1 = zeros(T);
            d_2 = zeros(T);
            for t = 1:T
                temp = zeros(length(epsilon_i{t}));
                temp_1 = zeros(length(epsilon_i{t}));
                temp_2 = zeros(length(epsilon_i{t}));
                for n = 1:length(epsilon_i{t})
                    temp(n) = (epsilon_i{t}(n))^2 - (epsilon_j{t}(n))^2;
                    temp_1(n) = epsilon_i{t}(n)^2;
                    temp_2(n) = epsilon_j{t}(n)^2;
                end
                d(t) = mean(temp(:));
                d_1(t) = mean(temp_1(:));
                d_2(t) = mean(temp_2(:));
            end
            
            % Compute bar_d
            bar_d = mean(d(:));

            % Compute s^2 and s
            s2 = var(d(:));
            s = sqrt(s2);

            % Compute z
            z = bar_d / (s / sqrt(T));

            p_value = 2 * (1 - normcdf(abs(z)));


            % Compute alpha_ij = normcdf(-z)
            alpha_ij = normcdf(-z);

            % significance_level = p_value;
            % if alpha_ij > 0.5
            %     significance = (1-significance_level) <= alpha_ij;
            % else
            %     significance = 1-significance_level <= 1-alpha_ij;
            % end
            significance = p_value < significance_level;
            % Store results
            bar_d_matrix_1(i,j) = mean(d_1(:));
            bar_d_matrix_2(i,j) = mean(d_2(:));
            bar_d_matrix(i,j) = bar_d;
            z_matrix(i,j)     = z;
            p_matrix(i,j)     = p_value;
            alpha_matrix(i,j) = alpha_ij;
            significance_matrix(i,j) = significance;
        end
    end
end
