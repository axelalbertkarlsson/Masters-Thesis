function [bar_d_matrix, z_matrix, p_matrix, alpha_matrix, significance_matrix] = model_comparason_likelihood(model_list, likelihoods, significance_level)
    % Number of models
    T = length(likelihoods.(model_list(1)));

    model_count = length(model_list);

    % Initialize results matrices
    bar_d_matrix = NaN(model_count, model_count);
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

            ell_i = likelihoods.(model_1);
            ell_j= likelihoods.(model_2);

            % Compute d_{i,k} via double loop
            d = zeros(T);
            for t = 1:T
                d(t) = (ell_i{t})^2 - (ell_j{t})^2;
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
            bar_d_matrix(i,j) = bar_d;
            z_matrix(i,j)     = z;
            p_matrix(i,j)     = p_value;
            alpha_matrix(i,j) = alpha_ij;
            significance_matrix(i,j) = significance;
        end
    end
end
