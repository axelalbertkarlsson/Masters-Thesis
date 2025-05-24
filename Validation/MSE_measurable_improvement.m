function [meas_improve_matrix] = MSE_measurable_improvement(model_list, bar_d_matrix_1, bar_d_matrix_2, alpha_matrix)
    % Number of models
   

    model_count = length(model_list);

    % Initialize results matrices
    meas_improve_matrix = NaN(model_count, model_count);

    % Loop through model pairs
    for i = 1:model_count
        for j = 1:model_count
            if i == j
                continue; % Skip same-model comparison
            end
            if alpha_matrix(i,j) >= 0.5
                d_j = bar_d_matrix_2(i,j) - bar_d_matrix_1(i,j);
            else
                d_j = 0;
            end
            %%%% ... noise but seems unneccesary

            meas_improve_matrix(i,j)= sqrt((2*d_j)/pi);


        end
    end
end
