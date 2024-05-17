%% Metrics for Comparison - Normalized Mean Square Error (NMSE), Mean Absolute Percentage Error (MAPE)
function [nmse, mape] = metrics(y, yhat)
    %% Input Arguments
    %   y: Actual Signal
    %   y_hat: Predicted Signal
    
    %% Output Arguments
    %   nmse: Normalized Mean Square Error
    %   mape: Mean Absolute Percentage Error
    
    %% Preprocess Signal
    % Ensure that y and y_hat are column vectors
    y = y(:);
    yhat = yhat(:);
    eps = 1e-10;
    % Remove Nan
    yhat = yhat(~isnan(y));
    y = y(~isnan(y));
    % Remove numerically insignificant values
    indices = y > eps;
    y = y(indices);
    yhat = yhat(indices);
    
    %% Compute Normalized Mean Square Error (NMSE)
    mse = mean((y - yhat).^2);  % Mean Square Error
    var_y = var(y);             % Variance of actual values
    nmse = mse / var_y;         % Normalized Mean Square Error
    
    %% Compute Mean Absolute Percentage Error (MAPE)
    % Compute the absolute errors
    abs_error = abs(y - yhat);
    % Compute the absolute values of y, excluding zeros
    abs_y = abs(y);
    nonzero_indices = abs_y > 0;
    % Compute the absolute percentage errors for non-zero y values
    ape = abs_error(nonzero_indices) ./ abs_y(nonzero_indices);
    mape = mean(ape) * 100; % Mean Absolute Percentage Error
end