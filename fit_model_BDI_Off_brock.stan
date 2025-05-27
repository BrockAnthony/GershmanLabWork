data {
  int<lower=1> Nsubject;                // Number of subjects
  int<lower=1> Ntrial;                  // Number of trials per subject
  matrix[Ntrial, Nsubject] stimulus;    // True stimulus values [trial, subject]
  matrix[Ntrial, Nsubject] estimate;    // Subject estimates [trial, subject]
  matrix[Ntrial, Nsubject] incentive;   // Incentive condition (0.4 or 2) [trial, subject]
  matrix[Ntrial, Nsubject] avg_stim;    // Average stimulus (trial-level) [trial, subject]
  array[Ntrial, Nsubject] int<lower=0, upper=1> valid;  
  vector[Nsubject] scaledBDI;         // Scaled BDI scores per subject
}

transformed data {
  matrix[Ntrial, Nsubject] log_stimulus;
  matrix[Ntrial, Nsubject] log_estimate;
  matrix[Ntrial, Nsubject] log_avg_stim;
  matrix[Ntrial, Nsubject] condition;      // Binary reward indicator (0 or 1)
  real lambda0 = 12/(30^2);             // Fixed prior precision
  
  // Calculate log transformations
  for (s in 1:Nsubject) {
    for (t in 1:Ntrial) {
      log_stimulus[t, s] = log(stimulus[t, s]);
      log_estimate[t, s] = log(estimate[t, s]);
      log_avg_stim[t, s] = log(avg_stim[t, s]);
      
      // Convert incentive to condition (0 - low incentive, 1 - high incentive)
      if (incentive[t, s] == 2.0) {
        condition[t, s] = 1.0;
      } else {
        condition[t, s] = 0.0;
      }
    }
  }
}

parameters {
  // Group-level parameters (excludes BDI effect Beta1)
  real<lower=0.0001, upper=2> mu_tau;     // Mean of base noise parameter
  real<lower=0.001, upper=25> mu_beta0;   // Mean baseline alpha
  real<lower=0, upper=15> mu_beta2;       // Mean reward condition effect
  
  // Standard deviations for subject-level random effects
  real<lower=0> sigma_tau;
  real<lower=0> sigma_beta0;
  real<lower=0> sigma_beta2;
  
  // Subject-level random effects
  vector[Nsubject] z_tau;
  vector[Nsubject] z_beta0;
  vector[Nsubject] z_beta2;
}

transformed parameters {
  // Subject-level parameters
  vector<lower=0.0001>[Nsubject] tau;
  vector<lower=0.001>[Nsubject] beta0;
  vector<lower=0>[Nsubject] beta2;
  
  // Trial-level parameters
  matrix[Ntrial, Nsubject] alpha;
  matrix[Ntrial, Nsubject] lambda;
  matrix[Ntrial, Nsubject] w;
  matrix[Ntrial, Nsubject] log_estimate_predicted;
  matrix[Ntrial, Nsubject] log_var;
  
  // Calculate subject-level parameters
  for (s in 1:Nsubject) {
    tau[s] = fmax(mu_tau + sigma_tau * z_tau[s], 0.0001);
    beta0[s] = fmax(mu_beta0 + sigma_beta0 * z_beta0[s], 0.001);
    beta2[s] = fmax(mu_beta2 + sigma_beta2 * z_beta2[s], 0.0);
    
    // Calculate trial-level parameters for this subject
    for (t in 1:Ntrial) {
      // Calculate alpha using regression model WITHOUT BDI: alpha = beta0 + beta2*condition
      // (equivalent to beta1 = 0 in the full model)
      alpha[t, s] = beta0[s] + beta2[s] * condition[t, s];
      
      // Ensure alpha is positive
      alpha[t, s] = fmax(alpha[t, s], 0.001);
      
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
  // Priors for group-level parameters
  mu_tau ~ uniform(0.0001, 2);
  mu_beta0 ~ uniform(0.001, 50);
  mu_beta2 ~ uniform(0, 15);
  
  // Priors for standard deviations
  sigma_tau ~ exponential(1);
  sigma_beta0 ~ exponential(0.5);
  sigma_beta2 ~ exponential(0.5);
  
  // Priors for random effects
  z_tau ~ normal(0, 1);
  z_beta0 ~ normal(0, 1);
  z_beta2 ~ normal(0, 1);
  
  // Likelihood
  for (s in 1:Nsubject) {
    for (t in 1:Ntrial) {
      if (valid[t, s] == 1) {  // Only include valid observations
        real sigma = sqrt(log_var[t, s]);
        target += normal_lpdf(log_estimate[t, s] | log_estimate_predicted[t, s], sigma);
      }
    }
  }
}

generated quantities {
  // Calculate effects (beta1/BDI effect is 0)
  real bdi_effect = 0.0;               // BDI effect is turned off
  real incentive_effect = mu_beta2;     // Main incentive effect
  
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

