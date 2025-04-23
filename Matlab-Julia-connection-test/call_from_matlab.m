% Ensure correct Python version
pe = pyenv;
if pe.Status == "NotLoaded" || ~contains(pe.Executable, "Python312")
    pyenv("Version", "C:\Users\axela\AppData\Local\Programs\Python\Python312\python.exe");
end

% Add wrapper path
if count(py.sys.path, 'julia') == 0
    insert(py.sys.path, int32(0), 'julia');
end

% Define input matrix
mat = [1.0, 2.0, 3.0; 4.0, 5.0, 7.0];

% Convert MATLAB matrix → Python list of lists
py_input = py.list(arrayfun(@(i) py.list(mat(i, :)), 1:size(mat,1), 'UniformOutput', false));

% Call Python-Julia function
res = py.gradient_wrapper.compute_gradients_and_loss_py(py_input);

% Extract losses
losses = double(py.list(res{"losses"}));

% Extract gradients
grads_py = res{"gradients"};
[n, m] = size(mat);
grads = zeros(n, m);
for i = 1:n
    for j = 1:m
        grads(i, j) = double(grads_py{i}{j});
    end
end

% Display result
disp('Losses:');
disp(losses);

disp('Gradients:');
disp(grads);