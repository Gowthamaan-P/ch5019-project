function [periods, powers] = lomb_scargle_periodogram(time, data)
    N = length(time);
    mean_data = mean(data);
    data = data - mean_data; % Subtract mean

    periods = linspace(2*max(time)/N, max(time), 1000); % Trial periods
    powers = zeros(size(periods));

    for i = 1:length(periods)
        freq = 2*pi/periods(i);
        coeffs = gradient_descent_fit(time, data, freq); % Estimate coefficients
        powers(i) = compute_lomb_scargle_power(time, data, freq, coeffs);
    end

    powers = powers / sum(powers);
end

function coeffs = gradient_descent_fit(time, data, freq)
    N = length(time);
    a0 = 0; b0 = 0; % Initial guesses
    learning_rate = 0.01;
    max_iter = 1000;

    for iter = 1:max_iter
        a_grad = 0; b_grad = 0;
        for i = 1:N
            err = data(i) - (a0*cos(freq*time(i)) + b0*sin(freq*time(i)));
            a_grad = a_grad - err * cos(freq*time(i));
            b_grad = b_grad - err * sin(freq*time(i));
        end
        a0 = a0 - learning_rate * a_grad/N;
        b0 = b0 - learning_rate * b_grad/N;
    end

    coeffs = [a0, b0];
end

function power = compute_lomb_scargle_power(time, data, freq, coeffs)
    N = length(time);
    a = coeffs(1); b = coeffs(2);
    mean_data = mean(data);
    data = data - mean_data; % Subtract mean

    c = sum(data .* cos(freq*time))/N;
    s = sum(data .* sin(freq*time))/N;
    cc = sum(cos(freq*time).^2)/N;
    ss = sum(sin(freq*time).^2)/N;
    cs = sum(cos(freq*time) .* sin(freq*time))/N;
    
    power = 0.5 * (c^2/cc + s^2/ss) / var(data);
end