%% R-Square
% This function calculates the R-squared (coefficient of determination)
function r_squared = rsquare(y, y_hat)
    %% Input Arguments
    %   y: Ground Truth
    %   y_hat: Predicted values

    %% Output Arguments
    %   r_squared: R-squared value (0 to 1)

    %% Criterion check and Initialize parameters
    %Check for equal vector lengths
    if length(y) ~= length(y_hat)
      error('Input vectors must have the same length');
    end
    % R-square is defined as the ratio of the sum of squares of the regression (SSR) and the total sum of squares (SST)
    % Calculate mean of actual values
    y_mean = mean(y);

    %% SST
    SST = sum((y - y_mean).^2);

    %% SSR
    SSR = sum((y - y_hat).^2);

    %% R-squared
    r_squared = 1 - SSR / SST;

    %% Handle potential division by zero
    if isnan(r_squared)
      r_squared = 0;
    end
end