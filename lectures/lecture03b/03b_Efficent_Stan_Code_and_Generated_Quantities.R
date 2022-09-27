# Package installation ======================================================================
needed_packages = c("invgamma", "ggplot2", "cmdstanr", "HDInterval", "bayesplot")
for(i in 1:length(needed_packages)){
  haspackage = require(needed_packages[i], character.only = TRUE)
  if(haspackage == FALSE){
    install.packages(needed_packages[i])
  }
  library(needed_packages[i], character.only = TRUE)
}

# Data import ==================================================================
DietData = read.csv(file = "DietData.csv")

# Centering variables ==========================================================
DietData$Height60IN = DietData$HeightIN-60

group2 = rep(0, nrow(DietData))
group2[which(DietData$DietGroup == 2)] = 1

group3 = rep(0, nrow(DietData))
group3[which(DietData$DietGroup == 3)] = 1

heightXgroup2 = DietData$Height60IN*group2
heightXgroup3 = DietData$Height60IN*group3


# adding prior to betas
model04_Syntax = "

data {
  int<lower=0> N;
  vector[N] weightLB;
  vector[N] height60IN;
  vector[N] group2;
  vector[N] group3;
  vector[N] heightXgroup2;
  vector[N] heightXgroup3;
}


parameters {
  real beta0;
  real betaHeight;
  real betaGroup2;
  real betaGroup3;
  real betaHxG2;
  real betaHxG3;
  real<lower=0> sigma;
}


model {
  beta0 ~ normal(0,1000);
  betaHeight ~ normal(0,1000);
  betaGroup2 ~ normal(0,1000);
  betaGroup3 ~ normal(0,1000);
  betaHxG2 ~ normal(0,1000);
  betaHxG3 ~ normal(0,1000);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

model04_Stan = cmdstan_model(stan_file = write_stan_file(model04_Syntax))

mode04_StanData = list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)


model04_Samples = model04_Stan$sample(
  data = mode04_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
model04_Samples$summary()


# making it easier to supply input into Stan ====================================

# adding prior to betas
model05_Syntax = "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;       // prior mean vector for coefficients
  matrix[P, P] covBeta; // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
}


parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}


model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

"

# start with model formula
model05_formula = formula(WeightLB ~ Height60IN + factor(DietGroup) + Height60IN:factor(DietGroup), data = DietData)

# grab model matrix
model05_predictorMatrix = model.matrix(model05_formula, data=DietData)
dim(model05_predictorMatrix)

# find details of model matrix
N = nrow(model05_predictorMatrix)
P = ncol(model05_predictorMatrix)

# build matrices of hyper parameters (for priors)
meanBeta = rep(0, P)
covBeta = diag(x = 10000, nrow = P, ncol = P)
sigmaRate = .1

# build Stan data from model matrix
model05_data = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate
)

model05_Stan = cmdstan_model(stan_file = write_stan_file(model05_Syntax))

model05_Samples = model05_Stan$sample(
  data = model05_data,
  seed = 23092022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
model05_Samples$summary()

# maximum R-hat
max(model05_Samples$summary()$rhat)

# Forming Diet Group 2 Slope ============

slopeG2 = model05_Samples$draws("beta[2]") + model05_Samples$draws("beta[4]")

summary(slopeG2)

# posterior histograms
mcmc_hist(slopeG2)

# posterior densities
mcmc_dens(slopeG2)

# Forming Diet Group 2 via Stan Generated Quantites (scalar version) ==========

# adding prior to betas
model04b_Syntax = "

data {
  int<lower=0> N;
  vector[N] weightLB;
  vector[N] height60IN;
  vector[N] group2;
  vector[N] group3;
  vector[N] heightXgroup2;
  vector[N] heightXgroup3;
}


parameters {
  real beta0;
  real betaHeight;
  real betaGroup2;
  real betaGroup3;
  real betaHxG2;
  real betaHxG3;
  real<lower=0> sigma;
}


model {
  beta0 ~ normal(0,1);
  betaHeight ~ normal(0,1000);
  betaGroup2 ~ normal(0,1000);
  betaGroup3 ~ normal(0,1000);
  betaHxG2 ~ normal(0,1000);
  betaHxG3 ~ normal(0,1000);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

generated quantities{
  real slopeG2;
  slopeG2 = betaHeight + betaHxG2;
}
"

model04b_Stan = cmdstan_model(stan_file = write_stan_file(model04b_Syntax))

mode04b_StanData = list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)


model04b_Samples = model04b_Stan$sample(
  data = mode04b_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
model04b_Samples$summary()

# Stan with Matrices and Contrasts ============================================

# adding prior to betas
model05b_Syntax = "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;     // prior mean vector for coefficients
  matrix[P, P] covBeta;   // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
  
  int<lower=0> nContrasts; 
  matrix[nContrasts,P] contrastMatrix;   // contrast matrix for additional effects
}


parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

generated quantities {
  vector[nContrasts] contrasts;
  contrasts = contrastMatrix*beta;
}
"

nContrasts = 2 
contrastMatrix = matrix(data = 0, nrow = nContrasts, ncol = P)
contrastMatrix[1,2] = contrastMatrix[1,5] = 1 # for slope for group=2
contrastMatrix[2,1] = contrastMatrix[2,3] = 1 # for intercept for group=2

# build Stan data from model matrix
model05b_data = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate,
  contrastMatrix = contrastMatrix,
  nContrasts = nContrasts
)

model05b_Stan = cmdstan_model(stan_file = write_stan_file(model05b_Syntax))

model05b_Samples = model05b_Stan$sample(
  data = model05b_data,
  seed = 23092022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
model05b_Samples$summary()

# assess convergence: summary of all parameters
max(model05b_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model05b_Samples$draws("contrasts[1]"))

# posterior histograms
mcmc_hist(model05b_Samples$draws("contrasts[1]"))

# posterior densities
mcmc_dens(model05b_Samples$draws("contrasts[1]"))


# Stan with Matrices, Contrasts, and R^2 ============================================

# adding prior to betas
model05c_Syntax = "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;     // prior mean vector for coefficients
  matrix[P, P] covBeta;   // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
  
  int<lower=0> nContrasts; 
  matrix[nContrasts,P] contrastMatrix;   // contrast matrix for additional effects
}


parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

generated quantities {
  vector[nContrasts] contrasts;
  contrasts = contrastMatrix*beta;
  
  real rss;
  real totalrss;
  {
    vector[N] pred;
    pred = X*beta;
    rss = dot_self(y-pred);
    totalrss = dot_self(y-mean(y));
  }
  
  real R2;
  R2 = 1-rss/totalrss;
}
"

nContrasts = 2 
contrastMatrix = matrix(data = 0, nrow = nContrasts, ncol = P)
contrastMatrix[1,2] = contrastMatrix[1,5] = 1 # for slope for group=2
contrastMatrix[2,1] = contrastMatrix[2,3] = 1 # for intercept for group=2

# build Stan data from model matrix
model05c_data = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate,
  contrastMatrix = contrastMatrix,
  nContrasts = nContrasts
)

model05c_Stan = cmdstan_model(stan_file = write_stan_file(model05c_Syntax))

model05c_Samples = model05c_Stan$sample(
  data = model05c_data,
  seed = 23092022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
model05c_Samples$summary()

# assess convergence: summary of all parameters
max(model05c_Samples$summary()$rhat, na.rm=TRUE)

# visualize posterior timeseries
mcmc_trace(model05c_Samples$draws("R2"))

# posterior histograms
mcmc_hist(model05c_Samples$draws("R2"))

# posterior densities
mcmc_dens(model05c_Samples$draws("R2"))


save.image(file = "03b.RData")
