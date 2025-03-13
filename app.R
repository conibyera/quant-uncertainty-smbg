#########################################################################
# This R Script use primary studies to derive the probability of SMBG    
# effectiveness versus fingerprick using the naive meta-analysis,
# the method of Wolpert et al/Eddy, and the method of Turner et al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
########################################################################

# Install required packages if not already installed:
# install.packages("rstan")
# install.packages("bayesplot")
# install.packages("loo")

library(rstan)
library(bayesplot)
library(loo)

# Set options for parallel computing in Stan (optional)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# -------------------------------
# 1. Prepare the Data
# -------------------------------

# Study data: reported mean difference (md) and 95% CI bounds.
studies <- data.frame(
  study = c("Beck (2017)", "Bergenstal (2010)", "Hermanides (2011)", 
            "Lind (2017)", "Rosenlund (2015)", "McQueen (2014)", "Soupal (2016)"),
  md = c(-0.60, -0.60, -1.45, -0.47, -0.70, -0.11, -1.50),
  lower = c(-0.86, -0.76, -1.63, -0.53, -1.23, -0.37, -1.84),
  upper = c(-0.36, -0.44, -1.27, -0.41, -0.17, 0.15, -1.16)
)

# Calculate standard errors using SE = (upper - lower)/3.92
studies$se <- (studies$upper - studies$lower) / 3.92

print(studies)

# Define bias information for each study.
# Each study has a list of bias priors (a, b) for the Beta distribution.
bias_list <- list(
  list(c(1,20), c(2,20)),               # Beck (2017): 2 biases
  list(c(1,20), c(2,20)),               # Bergenstal (2010): 2 biases
  list(c(1,20), c(2,20)),               # Hermanides (2011): 2 biases
  list(c(0.5,20), c(2,20)),             # Lind (2017): 2 biases
  list(c(0.5,20), c(2,20)),             # Rosenlund (2015): 2 biases
  list(c(0.5,20), c(2,20)),             # McQueen (2014): 2 biases
  list(c(0.5,20), c(2,20), c(0.5,25))    # Soupal (2016): 3 biases
)

N <- nrow(studies)
max_bias <- 3  # maximum number of bias types across studies

# Number of biases per study
n_bias <- sapply(bias_list, length)

# Create matrices for bias prior parameters (a and b)
a_bias <- matrix(NA, nrow=N, ncol=max_bias)
b_bias <- matrix(NA, nrow=N, ncol=max_bias)
for(i in 1:N){
  biases <- bias_list[[i]]
  n_i <- length(biases)
  for(j in 1:n_i){
    a_bias[i, j] <- biases[[j]][1]
    b_bias[i, j] <- biases[[j]][2]
  }
}

# Replace NA values with dummy values (here, 1). These will be ignored by Stan.
a_bias[is.na(a_bias)] <- 1
b_bias[is.na(b_bias)] <- 1

# Data list for the naive meta-analysis (no bias adjustment)
stan_data_naive <- list(
  N = N,
  md = studies$md,
  se = studies$se
)

# Data list for the bias-adjusted meta-analysis
stan_data_bias <- list(
  N = N,
  md = studies$md,
  se = studies$se,
  max_bias = max_bias,
  n_bias = as.integer(n_bias),
  a_bias = a_bias,
  b_bias = b_bias
)

# -------------------------------
# 2. Stan Model Codes
# -------------------------------

# a) Naïve Meta-Analysis Model (No Bias Adjustment)
naive_model_code <- "
data {
  int<lower=1> N;
  vector[N] md;
  vector<lower=0>[N] se;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta;
}
model {
  mu ~ normal(0, 10);
  tau ~ cauchy(0, 2);
  theta ~ normal(mu, tau);
  md ~ normal(theta, se);
}
generated quantities {
  // Posterior probability that mu < -0.4
  real prob_mu_lt_min;
  prob_mu_lt_min = step(-0.4 - mu);
}
"

# b) Bias-Adjusted Meta-Analysis Model (Wolpert et al.'s / Eddy's approach)
bias_model_code <- "
data {
  int<lower=1> N;
  vector[N] md;
  vector<lower=0>[N] se;
  int<lower=1> max_bias;
  int<lower=0> n_bias[N];
  matrix[N, max_bias] a_bias;
  matrix[N, max_bias] b_bias;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta;
  // Bias terms: one for each study and each potential bias, constrained between 0 and 1.
  matrix<lower=0, upper=1>[N, max_bias] bias;
}
model {
  mu ~ normal(0, 10);
  tau ~ cauchy(0, 2);
  theta ~ normal(mu, tau);
  // Loop over studies and biases
  for (i in 1:N) {
    for (j in 1:n_bias[i]) {\n      bias[i, j] ~ beta(a_bias[i, j], b_bias[i, j]);\n    }\n    // The observed effect is adjusted by subtracting the sum of bias terms\n    md[i] ~ normal(theta[i] - sum(bias[i, 1:n_bias[i]]), se[i]);\n  }\n}\n\ngenerated quantities {\n  real prob_mu_lt_min;\n  prob_mu_lt_min = step(-0.4 - mu);\n}\n"

# -------------------------------
# 3. Fit the Models Using rstan
# -------------------------------

# Compile the Stan models
naive_model <- stan_model(model_code = naive_model_code)
bias_model  <- stan_model(model_code = bias_model_code)

# Fit the naive model
fit_naive <- sampling(naive_model, data = stan_data_naive, 
                      iter = 4000, warmup = 1000, chains = 4, seed = 123)

# Fit the bias-adjusted model
fit_bias <- sampling(bias_model, data = stan_data_bias, 
                     iter = 4000, warmup = 1000, chains = 4, seed = 123)

# -------------------------------
# 4. Summarize and Compare Results
# -------------------------------

# Print summary for naive meta-analysis
print(fit_naive, pars = c("mu", "tau", "prob_mu_lt_min"))

# Print summary for bias-adjusted meta-analysis
print(fit_bias, pars = c("mu", "tau", "prob_mu_lt_min"))

# Extract posterior samples for mu
naive_extract <- extract(fit_naive)
bias_extract  <- extract(fit_bias)

mu_naive <- naive_extract$mu
mu_bias  <- bias_extract$mu

# Compute 95% uncertainty intervals (percentile intervals)
CI_naive <- quantile(mu_naive, probs = c(0.025, 0.975))
CI_bias  <- quantile(mu_bias, probs = c(0.025, 0.975))

cat("Naive Meta-analysis:\n")
cat("95% CI for mu: [", CI_naive[1], ",", CI_naive[2], "]\n")
cat("Posterior probability (mu < -0.4): ", mean(mu_naive < -0.4), "\n\n")

cat("Bias-Adjusted Meta-analysis:\n")
cat("95% CI for mu: [", CI_bias[1], ",", CI_bias[2], "]\n")
cat("Posterior probability (mu < -0.4): ", mean(mu_bias < -0.4), "\n")



#c) Bias-Adjusted Meta-Analysis Model (Turner et al.'s approach)

# Load necessary library
library(metafor)

# Step 1: Input observed effect estimates and their variances
# Data for each study: MD and 95% CI
studies <- data.frame(
  study = c("Beck (2017)", "Bergenstal (2010)", "Hermanides (2011)", 
            "Lind (2017)", "Rosenlund (2015)", "McQueen (2014)", "Soupal (2016)"),
  MD = c(-0.60, -0.60, -1.45, -0.47, -0.70, -0.11, -1.50),
  CI_lower = c(-0.86, -0.76, -1.63, -0.53, -1.23, -0.37, -1.84),
  CI_upper = c(-0.36, -0.44, -1.27, -0.41, -0.17, 0.15, -1.16)
)

# Calculate standard errors (SE = (upper - lower)/(2*1.96))
studies$SE <- (studies$CI_upper - studies$CI_lower) / (2 * qnorm(0.975))
# Calculate variances
studies$var <- studies$SE^2

# Step 2: Specify bias priors for each study.
# Each bias is represented by a Beta(a, b) distribution.
bias_priors <- list(
  list(performance = c(1, 20), detection = c(2, 20)),
  list(performance = c(1, 20), detection = c(2, 20)),
  list(performance = c(1, 20), detection = c(2, 20)),
  list(performance = c(0.5, 20), detection = c(2, 20)),
  list(performance = c(0.5, 20), detection = c(2, 20)),
  list(detection = c(0.5, 20), confounding = c(2, 20)),
  list(detection = c(0.5, 20), confounding = c(2, 20), analytic = c(0.5, 25))
)

# Step 3: Function to compute mean and variance from Beta(a, b)
bias_mean_var <- function(a, b) {
  m <- a / (a + b)
  v <- (a * b) / (((a + b)^2) * (a + b + 1))
  return(c(m, v))
}

# Initialize vectors for adjusted effect sizes and variances
y_adj <- numeric(nrow(studies))
v_adj <- numeric(nrow(studies))

# Step 4: Process each study to adjust for bias
for (i in 1:nrow(studies)) {
  # Observed effect size and variance for study i
  y_obs <- studies$MD[i]
  v_obs <- studies$var[i]
  
  total_bias_mean <- 0
  total_bias_var <- 0
  
  # Loop over each bias in study i
  biases <- bias_priors[[i]]
  if (length(biases) > 0) {
    for (bias_name in names(biases)) {
      a <- biases[[bias_name]][1]
      b <- biases[[bias_name]][2]
      stats <- bias_mean_var(a, b)
      bias_mean <- stats[1]  # first element
      bias_var  <- stats[2]  # second element
      
      # Sum biases (here, we subtract bias effect from observed MD)
      total_bias_mean <- total_bias_mean + bias_mean
      total_bias_var <- total_bias_var + bias_var
    }
  }
  # Adjust effect size and variance:
  y_adj[i] <- y_obs - total_bias_mean
  v_adj[i] <- v_obs + total_bias_var
}

# Step 5: Perform a random-effects meta-analysis on the bias-adjusted estimates
res <- rma(yi = y_adj, vi = v_adj, method = "REML")

# Display meta-analysis results
print(res)

# Step 6: Compute 95% uncertainty interval for the pooled estimate
pooled_estimate <- res$b
pooled_variance <- res$tau2 + res$se^2
ci_lower <- pooled_estimate - qnorm(0.975) * sqrt(pooled_variance)
ci_upper <- pooled_estimate + qnorm(0.975) * sqrt(pooled_variance)
cat("95% Uncertainty Interval: [", ci_lower, ",", ci_upper, "]\n")

# Step 7: Compute the probability that the pooled estimate is less than -0.4
prob_less_than_neg0.4 <- pnorm(-0.4, mean = pooled_estimate, sd = sqrt(pooled_variance))
cat("Probability that the pooled estimate is less than -0.4: ", prob_less_than_neg0.4, "\n")

















