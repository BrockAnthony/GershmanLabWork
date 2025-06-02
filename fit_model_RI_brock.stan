data {
  int<lower=1> Nsubject;              // Number of subjects
  int<lower=1> Ntrial;                // Number of trials per subject
  matrix[Ntrial, Nsubject] stimulus;  // True stimulus values [trial, subject]
  matrix[Ntrial, Nsubject] estimate;  // Subject estimates [trial, subject]
  matrix[Ntrial, Nsubject] incentive; // Incentive condition (0.4 or 2) [trial, subject]
  matrix[Ntrial, Nsubject] avg_stim;  // Average stimulus (trial-level) [trial, subject]
  array[Ntrial, Nsubject] int<lower=0, upper=1> valid;
}

transformed data {
  matrix[Ntrial, Nsubject] log_stimulus;
  matrix[Ntrial, Nsubject] log_estimate;
  matrix[Ntrial, Nsubject] log_avg_stim;
  real lambda0 = 12/(30^2);          // Fixed prior precision
  
  // Calculate log transformations
  for (s in 1:Nsubject) {
    for (t in 1:Ntrial) {
      log_stimulus[t, s] = log(stimulus[t, s]);
      log_estimate[t, s] = log(estimate[t, s]);
      log_avg_stim[t, s] = log(avg_stim[t, s]);
    }
  }
}

parameters {
  // Group-level parameters
  real<lower=0.0001, upper=2> mu_tau; // Mean of base noise parameter
  real<lower=0.0001, upper=50> mu_alpha_low;  // Mean alpha for low incentive
  real<lower=0.0001, upper=50> mu_alpha_high; // Mean alpha for high incentive
  
  // Standard deviations for subject-level random effects
  real<lower=0> sigma_tau;
  real<lower=0> sigma_alpha_low;
  real<lower=0> sigma_alpha_high;
  
  // Subject-level random effects (non-centered parameterization)
  vector[Nsubject] z_tau;
  vector[Nsubject] z_alpha_low;
  vector[Nsubject] z_alpha_high;
}

transformed parameters {
  // Subject-level parameters
  vector<lower=0.0001>[Nsubject] tau;
  vector<lower=0.0001>[Nsubject] alpha_low;
  vector<lower=0.0001>[Nsubject] alpha_high;
  
  // Trial-level parameters
  matrix[Ntrial, Nsubject] alpha;
  matrix[Ntrial, Nsubject] lambda;
  matrix[Ntrial, Nsubject] w;
  matrix[Ntrial, Nsubject] log_estimate_predicted;
  matrix[Ntrial, Nsubject] log_var;
  
  // Calculate subject-level parameters
  for (s in 1:Nsubject) {
    tau[s] = fmax(mu_tau + sigma_tau * z_tau[s], 0.0001);
    alpha_low[s] = fmax(mu_alpha_low + sigma_alpha_low * z_alpha_low[s], 0.0001);
    alpha_high[s] = fmax(mu_alpha_high + sigma_alpha_high * z_alpha_high[s], 0.0001);
    
    // Calculate trial-level parameters for this subject
    for (t in 1:Ntrial) {
      // Assign alpha based on incentive condition
      if (incentive[t, s] == 0.4) {
        alpha[t, s] = alpha_low[s];
      } else {
        alpha[t, s] = alpha_high[s];
      }
      
      // Calculate precision
      lambda[t, s] = fmax(2 * alpha[t, s] - lambda0, 0.0001);
      
      // Calculate Bayesian weight
      w[t, s] = lambda[t, s] / (lambda[t, s] + lambda0);
      
      // Calculate predicted log-estimate
      log_estimate_predicted[t, s] = w[t, s] * log_stimulus[t, s] + (1 - w[t, s]) * log_avg_stim[t, s];
      
      // Calculate predicted variance of log-estimate
      log_var[t, s] = ((w[t, s]^2) / lambda[t, s]) + tau[s];
    }
  }
}

model {
  mu_tau ~ cauchy(0, 0.5);           
  mu_alpha_low ~ cauchy(0, 10);   
  mu_alpha_high ~ cauchy(0, 10); 
  
  sigma_tau ~ exponential(1);
  sigma_alpha_low ~ exponential(0.5);
  sigma_alpha_high ~ exponential(0.5);
  
  // Priors for random effects
  z_tau ~ normal(0, 1);
  z_alpha_low ~ normal(0, 1);
  z_alpha_high ~ normal(0, 1);
  
  // Likelihood
  for (s in 1:Nsubject) {
    for (t in 1:Ntrial) {
      if (valid[t, s] == 1) { 
        real sigma = sqrt(log_var[t, s]);
        target += normal_lpdf(log_estimate[t, s] | log_estimate_predicted[t, s], sigma);
      }
    }
  }
}

generated quantities {
  // Calculate difference parameter for high vs low incentive
  real incentive_effect = mu_alpha_high - mu_alpha_low;
  
  // Calculate confidence and log likelihood for each trial
  matrix[Ntrial, Nsubject] confidence;
  matrix[Ntrial, Nsubject] log_lik;
  
  // Calculate trial-level quantities
  for (s in 1:Nsubject) {
    for (t in 1:Ntrial) {
      // Calculate confidence
      confidence[t, s] = 1 / (lambda[t, s] + lambda0);
      
      // Calculate log likelihood
      if (valid[t, s] == 1) {
        real sigma = sqrt(log_var[t, s]);
        log_lik[t, s] = normal_lpdf(log_estimate[t, s] | log_estimate_predicted[t, s], sigma);
      } else {
        log_lik[t, s] = 0;  // Set to 0 for invalid trials
      }
    }
  }
}

