%% Question - 1
% Clear workspace and command window
clear; clc;close all;
% For reproducibility
rng(13); 

%% Given Data
N = 20;  % Number of Observations
R = 200; % Number of Realizations
beta = [0;3;5]; % True Beta

%% Estimation Method
method = 'Ordinary Least Squares';
fprintf(['<strong>Linear Regression with ', method ,'</strong>\n\n']);
%% Initialize variables
beta_estimates = zeros(R, 3); % Store parameter estimates
loss = zeros(R, 1); % Store final cost
alpha = 0.003; % Learning rate
epochs = 1000; % Epochs or Number of Iterations

%% Perform Regression for R realizations
for r = 1:R
    % Generate Data
    X1 = randn(N, 1);
    X2 = randn(N, 1);
    E = randn(N, 1);
    y = beta(2)*X1 + beta(3)*X2 + E;
    X = [X1 X2];
    
    % Perform Regression
    [beta_hat, cost_history] = ols(X, y, alpha, epochs);
    beta_estimates(r, :) = beta_hat';
    loss(r) = cost_history(end);
end

%% Best beta across realizations
% Find the realization with the minimum sum of squared residuals
[min_ssr, best_parameter] = min(loss);
fprintf('Best realization Index: %d\n\n', best_parameter);
fprintf('Best parameter estimates: ');
disp(beta_estimates(best_parameter, :));
fprintf('Loss: %.4f\n\n', min_ssr);


%% Plot Results
% Generate Test Data
X1 = randn(N, 1);
X2 = randn(N, 1);
E = randn(N, 1);
y = beta(2)*X1 + beta(3)*X2 + E;
X = [X1 X2];
% Plot Results
plotresult(X, y, beta_estimates(best_parameter, :)', epochs, cost_history, "Ordinary Least Squares")

%% Calculate Metrics
[mse, rb, mad] = metrics(beta, beta_estimates);
% Model parameter names
parameters = {'beta0', 'beta1', 'beta2'};
% Display metrics
fprintf('Metrics\n\n');
result_table = table(parameters', mse', rb', mad', 'VariableNames', {'Parameters', 'MSE', 'RB', 'MAD'});
disp(result_table);