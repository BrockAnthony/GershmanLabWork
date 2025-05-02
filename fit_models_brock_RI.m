function [results] = fit_models_brock_RI(data, results)
    % Fit ONLY the Rational Inattention (RI) model to data using maximum 
    % likelihood estimation. 
    %
    % INPUTS:
    %   data    - a struct array, one element per subject,
    %             fields must include:
    %               .subject, .estimate, .stimulus, .incentive
    %             (and anything else the likelihood function needs)
    % 
    %   results (optional) existing results structure to modify
    %
    % OUTPUTS:
    %   results - structure array with fitted model parameters
    %
    % If 'results' is not provided, initialize as empty struct.
    if nargin < 2 || isempty(results)
        results = struct('param', [], 'likfun', [], 'x', [], ...
            'logpost', [], 'loglik', [], 'bic', [], 'aic', [], 'H', [], 'latents', []);
    end

    % Convert data struct array (load_data)
    clear D
    for s = 1:length(data)
        D(s).subject         = data(s).subject;      % subject ID
        D(s).est_stim        = data(s).estimate;     % estimates
        D(s).true_stim       = data(s).stimulus;     % true stimulus
        D(s).cond            = data(s).incentive;    % condition 

        D(s).log_estimate    = log(data(s).estimate);
        D(s).log_stimulus    = log(data(s).stimulus);

        D(s).avg_stim        = data(s).avg_stim; 
        D(s).log_avg_stim    = log(data(s).avg_stim);

        D(s).N               = length(data(s).estimate);  % number of trials
    end

    % fit RI model with mfit_optimize
    likfun = @(x,data) likfun_RI_inline(x,data); 
    % RI model parameters (ranges):
    %   tau         
    %   alpha_low  
    %   alpha_high 

    tau_range = 0.0001:0.005:0.02;           
    alpha_low_range = 0.0001:0.05:42;  
    alpha_high_range = 0.0001:0.05:42; 
    
    % Create parameter definitions with these ranges
    param(1) = struct('name', 'tau', 'range', tau_range, 'lb', min(tau_range), 'ub', max(tau_range));
    param(2) = struct('name', 'alpha_low', 'range', alpha_low_range, 'lb', min(alpha_low_range), 'ub', max(alpha_low_range));
    param(3) = struct('name', 'alpha_high', 'range', alpha_high_range, 'lb', min(alpha_high_range), 'ub', max(alpha_high_range));

    results = mfit_optimize(likfun, param, D);

    % Likelihood Function for the RI model
    function [lik, latents] = likfun_RI_inline(x, data)
        tau         = x(1);
        alpha_low   = x(2); 
        alpha_high  = x(3);

        lambda0 = 12/(30^2);

        % Assign alpha depending on condition
        alpha = zeros(size(data.cond)) + alpha_high;  
        alpha(double(data.cond) == 0.4) = alpha_low;     

        % Convert alpha into precision 
        lambda = max(2.*alpha - lambda0,0.0001);

        % Bayesian weight
        w = lambda./(lambda + lambda0);

        % Predicted log-estimate
        log_estimate_predicted = w.*data.log_stimulus + (1 - w).*data.log_avg_stim;

        % Predicted variance of log-estimate
        log_var = ((w.^2)./lambda) + tau;

        % % Valid trials (exclude any with NaNs or negative variance)
        % validIdx = ~isnan(data.log_estimate) & ...
        %            ~isnan(log_estimate_predicted) & ...
        %            (log_var > 0);

        % log-likelihood
        lik = sum(lognormpdf(data.log_estimate, log_estimate_predicted, sqrt(log_var)));

        % If latents requested, store predicted estimates, etc.
        if nargout > 1
            latents.log_estimate = log_estimate_predicted;
            latents.confidence   = 1./(lambda + lambda0);
            latents.log_var = log_var;
        end
    end
end
