function [results] = fit_models_brock_BDI_intact(data, BDI_intact)
    % Fit the Rational Inattention (RI) model to data using maximum 
    % likelihood estimation with an option to "turn off" the BDI effect.
    % If BDI_intact is set to 0, the Beta1 parameter is set to 0,
    % effectively removing the influence of BDI on the model.
    % If BDI_intact is 1 (or not provided), the model runs as normally defined.
    %
    % USAGE: [results] = fit_models_brock_BDI_intact(data, BDI_intact)
    %
    % INPUTS:
    %   data       - a struct array, one element per subject, fields must include:
    %                  .subject, .estimate, .stimulus, .incentive, .avg_stim, .survey.BDI
    % 
    %   BDI_intact - flag (0 or 1). If 0, Beta1 is forced to 0.
    %                (Default is 1.)
    %
    % OUTPUTS:
    %   results    - structure array with fitted model parameters

    if nargin < 2
        BDI_intact = 1;  % default: use BDI normally
    end

    % Initialize results structure if not provided
    results = struct('param', [], 'likfun', [], 'x', [], ...
            'logpost', [], 'loglik', [], 'bic', [], 'aic', [], 'H', [], 'latents', []);

    % Convert data struct array
    clear D
    for s = 1:length(data)
        D(s).subject      = data(s).subject;      % subject ID
        D(s).est_stim     = data(s).estimate;       % estimates (vector)
        D(s).true_stim    = data(s).stimulus;       % true stimulus (vector)

        D(s).log_estimate = log(data(s).estimate);
        D(s).log_stimulus = log(data(s).stimulus);

        D(s).avg_stim     = data(s).avg_stim; 
        D(s).log_avg_stim = log(data(s).avg_stim);

        D(s).N            = length(data(s).estimate);  % number of trials
        D(s).BDI          = data(s).scaled_BDI;


        % Convert incentive to R (0 for low; 1 for high incentives)
        if isnumeric(data(s).incentive)
            R = nan(size(data(s).incentive));  
            R(data(s).incentive == 2) = 1;
            R(data(s).incentive == 0.4) = 0;
            if any(isnan(R))
                error('Unknown incentive value for subject %d', s);
            end
            D(s).cond = R;
        else
            error('Unexpected data type for incentive in subject %d', s);
        end
    end

    likfun = @(x, data) likfun_RI_inline(x, data, BDI_intact); 

    % Define RI model parameters:
    %   tau 
    %   beta0 (baseline alpha)
    %   beta1  (effect of BDI score)
    %   beta2  (effect of reward condition)

    tau_range = 0.0001:0.001:2;        
    beta0_range = 0.001:0.005:50;             
    beta1_range = -25:0.001:25;             
    beta2_range = -25:0.005:25;                
    
    param(1) = struct('name', 'tau', 'range', tau_range, 'lb', min(tau_range), 'ub', max(tau_range));
    param(2) = struct('name', 'beta0', 'range', beta0_range, 'lb', min(beta0_range), 'ub', max(beta0_range));
    param(3) = struct('name', 'beta1', 'range', beta1_range, 'lb', min(beta1_range), 'ub', max(beta1_range));
    param(4) = struct('name', 'beta2', 'range', beta2_range, 'lb', min(beta2_range), 'ub', max(beta2_range));

    results = mfit_optimize(likfun, param, D);

    % Inline Likelihood Function for the RI model
    function [lik, latents] = likfun_RI_inline(x, data, BDI_intact)
        % Extracting parameters
        tau   = x(1);
        beta0 = x(2);
        beta1 = x(3);
        beta2 = x(4);
        
        % If BDI_intact is 0, force beta1 to 0 (ignore BDI)
        if BDI_intact == 0
            beta1 = 0;
        end

        lambda0 = 12/(30^2);

        % Compute alpha on a per-trial basis:
        alpha = beta0 + beta1 * data.BDI + beta2 * data.cond;

        % Convert alpha into precision 
        lambda = max(2.*alpha - lambda0, 0.0001);

        % Bayesian weight
        w = lambda ./ (lambda + lambda0);

        % Predicted log-estimate
        log_estimate_predicted = w .* data.log_stimulus + (1 - w) .* data.log_avg_stim;

        % Predicted variance of log-estimate
        log_var = ((w.^2)./lambda) + tau;

        % log-likelihood
        ll = lognormpdf(data.log_estimate, log_estimate_predicted, sqrt(log_var));

        ll(isnan(ll) | isinf(ll)) = -1000;

        lik = sum(ll);

        % If latents requested, store predicted estimates, etc.
        if nargout > 1
            latents.log_estimate = log_estimate_predicted;
            latents.confidence   = 1./(lambda + lambda0);
            latents.log_var = log_var;
        end
    end
end
