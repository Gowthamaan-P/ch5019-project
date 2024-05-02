%% Least Trimmed Squares using Gradient Descent
function [beta, cost_history] = lts(X, y, alpha, epochs)
    %% Input Arguments
    %   X: Input features (n x m matrix, where n is the number of data points
    %      and m is the number of features)
    %   y: Target variable (n x 1 vector)
    %   alpha: Learning rate (scalar)
    %   epochs: Number of epochs (scalar)

    %% Output Arguments
    %   beta: Learned or Estimated parameters of the model(m+1 x 1 vector) 
    %   cost_history: Cost history for each iteration (epochs x 1 vector)

    %% Initialize parameters
    n = length(y);                   % Number of data points
    q = floor((n/2)+1);              % Number of samples to consider for computing loss
    m = size(X, 2);                  % Number of features
    beta = zeros(m + 1, 1);         % Initialize parameters
    cost_history = zeros(epochs, 1); % Store cost for each iteration
    X = [ones(n, 1) X];              % Add bias term to X

    %% Gradient Descent
    for epoch = 1:epochs
        % Compute y_hat and cost
        y_hat = X * beta;              % Hypothesis: y_hat = X * theta
        e = y_hat - y;                 % Residual
        index_column = (1:n)';
        e_indexed = [e.^2, index_column]; % Add index column (Useful while calculating the gradients)
        e_sorted = sortrows(e_indexed, 1); % Ordered squared residuals 
        e_sampled = e_sorted(1:q, :); % Consider the first q samples
        cost = sum(e_sampled(:, 1))/ (2 * q); % Least Trimmed Squares
        cost_history(epoch) = cost; % Store cost for plotting
        
        % Compute gradients
        Xs = X(e_sampled(:, 2),:); % Sampled Input
        es = e(e_sampled(:, 2)); % Sampled Error
        grad = (Xs' * (es)) / q;

        % Update parameters
        beta = beta - alpha * grad;
    end
end