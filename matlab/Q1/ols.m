%% Ordinary Least Squares using Gradient Descent
function [beta, cost_history] = ols(X, y, alpha, epochs)
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
    m = size(X, 2);                  % Number of features
    beta = zeros(m + 1, 1);         % Initialize parameters
    cost_history = zeros(epochs, 1); % Store cost for each iteration
    X = [ones(n, 1) X];              % Add bias term to X

    %% Gradient Descent
    for epoch = 1:epochs
        % Compute y_hat and cost
        y_hat = X * beta;              % Hypothesis: y_hat = X * theta
        cost = sum((y_hat - y).^2) / (2 * n); % Ordinary Least Squares
        cost_history(epoch) = cost; % Store cost for plotting

        % Compute gradients
        grad = (X' * (y_hat - y)) / n;

        % Update parameters
        beta = beta - alpha * grad;
    end
end