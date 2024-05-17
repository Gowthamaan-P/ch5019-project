%% Lomb Scale Periodogram using Gradient Descent
function [frequencies, powers, cost_history, a, b] = lomb_scale_periodogram(time, signal, alpha, epochs, varargin)
    %% Input Arguments
    %   time: Time instances of the signal
    %   signal: Signal sampled at the instants specified in time
    %   alpha: Learning rate (scalar)
    %   epochs: Number of epochs (scalar)

    %% Output Arguments
    %   Returns the power spectral density estimate
    %       frequencies: List of frequncies
    %       powers: Powers at each frequency
    %   cost_history: Cost at every epoch using the LS method
    %   LS Periodogram Estimates
    %       a: Coefficients of cosine term 
    %       b: Coefficients of sine term
    
    %% Initialize variables
    N = length(time); % Length of the signal
    if ~isempty(varargin)
        K = varargin{1}; % User-defined Resolution of the frequency range considered for analysis
    else
        K = 1000; % Resolution of the frequency range considered for analysis
    end
    % Compute the frequency range for analysis
    lower_bound = 0; % Lower bound
    upper_bound = floor((N/time(end))/2); % Upper bound (Fs/2)
    frequencies = lower_bound:(upper_bound/K):upper_bound; % Construct frequency range
    m = length(frequencies); % Number of frequencies in the analysis range
    powers = zeros(m, 1); % Power of each signal in the frequency range
    a = zeros(m, 1); % Coefficient for cosine term
    b = zeros(m, 1); % Coefficient for sine term
    cost_history = zeros(epochs, 1); % Cost history
    
    %% Pre-process signal
    missing_indices = isnan(signal); % Identify the indices of missing data
    non_missing_signal = signal(~missing_indices); % Construct the signal omitting the missing indices
    mean_value = mean(non_missing_signal); % Compute mean
    centered_non_missing_signal = non_missing_signal - mean_value; % Mean center the signal
    % Processed signal
    y = signal;
    y(~missing_indices) = centered_non_missing_signal; %For non-missing indice,use the mean centered value
    y(missing_indices) = mean_value; % For missing indices, use the mean value
    
    %% Gradient Descent
    for epoch = 1:epochs
        % Compute y_hat and cost
        y_hat = zeros(N, 1);
        for i = 1:m
           w = 2*pi*frequencies(i); % w=2?f
           y_hat = y_hat + (a(i)*cos(w*time) + b(i)*sin(w*time)); % construct y_hat
        end
        residual = y-y_hat; % Residual
        cost_history(epoch) = (sum(residual.^2)/2*N); % Oridinary Least Squares
        
        % Compute gradients
        for i = 1:m
            w = 2*pi*frequencies(i);
            grad_a = -sum(residual.*cos(w*time))/N; % Gradient for a
            grad_b = -sum(residual.*sin(w*time))/N; % Gradient for b
            
            % Parameter updates
            a(i) = a(i) - alpha*grad_a;
            b(i) = b(i) - alpha*grad_b;
        end
    end
    
    %% Compute powers of each signal frequency
    for i = 1:m
        w = 2*pi*frequencies(i);
        s = a(i)*cos(w*time) + b(i)*sin(w*time); 
        powers(i) = sum(s.^2) / N;
    end
end