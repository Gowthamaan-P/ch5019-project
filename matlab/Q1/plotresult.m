%% Plot Regression Reults
function plotresult(X, y, beta, epochs, cost_history, method)
    %% Input Arguments
    %   X            - Input features (n x m matrix, where n is the number of
    %                  data points and m is the number of features)
    %   y            - Target variable (n x 1 vector)
    %   beta        - Learned parameters (m+1 x 1 vector, including bias term)
    %   epochs       - Number of iterations or epochs (scalar)
    %   cost_history - Cost for each iteration or epoch (epochs x 1 vector)
    %   method       - String describing the method used for regression

    %% Output Arguments
    %   NIL

    %% Cost function convergence plot
    figure;
    plot(1:epochs, cost_history, 'b-', 'LineWidth', 1);
    xlabel('Iteration', 'FontSize', 14);
    ylabel('Cost', 'FontSize', 14);
    title('Cost Function Convergence ('+ method + ')', 'FontSize', 14);
    set(gca, 'FontSize', 12); % Set font size for axis labels and ticks

    %% Linear regression model plot
    n = length(y);
    figure;
    scatter(1:n, y, 'rx', 'MarkerFaceColor', 'r'); % Exclude bias term from X
    hold on;
    X = [ones(n, 1) X];              % Add bias term to X
    plot(1:n, X*beta, 'b-', 'LineWidth', 1);
    legend('Test Data', 'Model Prediction', 'Location', 'NorthWest');
    xlabel('Input (X)', 'FontSize', 14);
    ylabel('Output (y)', 'FontSize', 14);
    title('Linear Regression Model ('+ method + ')', 'FontSize', 14);
    set(gca, 'FontSize', 12); % Set font size for axis labels and ticks
    hold off;
end