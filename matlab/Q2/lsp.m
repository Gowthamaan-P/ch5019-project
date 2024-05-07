function [power, frequency] = lsp(t, data, learning_rate, epochs)
    N = length(t);
    
    % Subtract mean (Mean-centered signal) 
    % Identify missing values
    missing_indices = isnan(data);
    % Remove missing values
    non_missing_signal = data(~missing_indices);
    % Compute the mean of non-missing values
    mean_value = mean(non_missing_signal);
    % Mean-center the non-missing values
    centered_non_missing_signal = non_missing_signal - mean_value;
    % Replace missing values with NaN (or any desired value)
    y = data;
    y(~missing_indices) = centered_non_missing_signal;


    % Initialize frequencies to search
    max_frequency = N/t(end); % Nyquist frequency
    frequency = linspace(0, max_frequency, N);
    
    % Initialize variables
    power = zeros(size(frequency));
    a = ones(size(frequency)); % Initial guess for a
    b = ones(size(frequency)); % Initial guess for b
    
    for i = 1:length(frequency)
        for epoch = 1:epochs
            w = 2*pi*frequency(i);
            y_hat = a(i)*cos(w*t) + b(i)*sin(w*t);
            residual = y-y_hat;
            
            power(i) = sum(y_hat.^2) / length(y_hat);
            
            grad_a = -2 * sum(residual.*cos(w*t));
            grad_b = -2 * sum(residual.*sin(w*t));
            
            a(i) = a(i) - learning_rate * grad_a;
            b(i) = b(i) - learning_rate * grad_b;
        end
    end
end