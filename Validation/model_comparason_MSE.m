function [bar_d1, bar_d2, bar_d, ...
    z_mat, p_mat, alpha_mat, sig_mat] = model_comparason_MSE(model_list, innovations, significance_level, index)
    % Number of models
    T = size(innovations,1);

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
        model1 = char(model_list(i));
        for j = 1:model_count
            if i == j
                continue; % Skip same-model comparison
            end
             model2     = char(model_list(j));
            
            eps_i = innovations(:,i);  % vector length T
            eps_j = innovations(:,j);
            d   = eps_i.^2 - eps_j.^2;     % T×1
            d1  = eps_i.^2;
            d2  = eps_j.^2;

                    % Sample statistics
              bar_dij = mean(d);
              s       = std(d);              % uses N–1
              z       = bar_dij / (s/sqrt(T));
              p_val   = 2*(1 - normcdf(abs(z)));
              aij     = normcdf(-z);
              sig     = (p_val < significance_level);
              
                    % Store
              bar_d1(i,j)   = mean(d1);
              bar_d2(i,j)   = mean(d2);
              bar_d(i,j)    = bar_dij;
              z_mat(i,j)    = z;
              p_mat(i,j)    = p_val;
              alpha_mat(i,j)= aij;
              sig_mat(i,j)  = sig;

                        % print them along with the model names
              fprintf(...
          'Sum of %s innovations = %.3g; Sum of %s innovations = %.3g; index = %d\n', ...
           model1, bar_d1(i,j), model2, bar_d2(i,j), index);
            
        end
    end
end
