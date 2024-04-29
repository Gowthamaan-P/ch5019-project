%% Metrics for Comparison - Mean Square Error (MSE), Robust Bias (RB), and Median Absolute Deviation (MAD)
function [mse, rb, mad] = metrics(beta, beta_estimates)
    %% Input Arguments
    %   beta (1xp double): A row vector containing the true parameter values.
    %   beta_estimates (nxp double): A matrix where each row represents a set of parameter estimates.
    %                           The number of columns should match the length of beta.
    
    %% Output Arguments
    %   mse (1xp double): A row vector containing the Mean Square Error for each parameter.
    %   rb (1xp double): A row vector containing the Robust Bias for each parameter.
    %   mad (1xp double): A row vector containing the Median Absolute Deviation for each parameter.
    
    %% Initialize arrays to store metrics
    mse = zeros(1, size(beta_estimates, 2));
    rb = zeros(1, size(beta_estimates, 2));
    mad = zeros(1, size(beta_estimates, 2));

    %% Calculate metrics for every parameter
    for parameter = 1:size(beta_estimates, 2)
        bias = mean(beta_estimates(:, parameter));
        variance = var(beta_estimates(:, parameter));
        mse(parameter) = bias^2 + variance;  % Mean Square Error
        rb(parameter) = median(beta_estimates(:, parameter)) - beta(parameter);  % Robust Bias
        mad(parameter) = median(abs(beta_estimates(:, parameter) - beta(parameter)));  % Median Absolute Deviation
    end
end