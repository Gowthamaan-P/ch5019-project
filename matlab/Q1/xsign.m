%% Sign Function
% This function calculates the element-wise sign of the input vector 'x'
% with a threshold value 'm'.
function sgn = xsign(x, threshold)
    %% Input Arguments
    %   x(1xn num): A vector of numerical values.
    %   threshold(num): A threshold value (optional). Defaults to zero.

    %% Output Arguments
    %   sgn: A vector containing the signs of the elements in 'x' relative 
    %        to the threshold.

    %% Calculate sign
    if nargin < 2  % Check if optional argument 'threshold' is provided
      sgn = 2 * (x > 0) - 1;  % Default case: sign based on zero threshold
    else
      sgn = 2 * (x > threshold) - 1;  % Sign based on provided threshold
    end
end