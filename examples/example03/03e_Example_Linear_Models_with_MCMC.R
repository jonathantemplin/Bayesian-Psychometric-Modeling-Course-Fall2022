# Package installation ======================================================================
needed_packages = c("invgamma", "ggplot2", "cmdstanr", "HDInterval", "bayesplot")
for(i in 1:length(needed_packages)){
  haspackage = require(needed_packages[i], character.only = TRUE)
  if(haspackage == FALSE){
    install.packages(needed_packages[i])
  }
  library(needed_packages[i], character.only = TRUE)
}

if (!requireNamespace("modeest")) install.packages("modeest")
library(modeest)


# Data import ==================================================================
DietData = read.csv(file = "DietData.csv")

ggplot(data = DietData, aes(x = WeightLB)) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 


ggplot(data = DietData, aes(x = WeightLB, color = factor(DietGroup), fill = factor(DietGroup))) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 


ggplot(data = DietData, aes(x = HeightIN, y = WeightLB, shape = factor(DietGroup), color = factor(DietGroup))) +
  geom_smooth(method = "lm", se = FALSE) + geom_point()

# Centering variables ==========================================================
DietData$Height60IN = DietData$HeightIN-60

# full analysis model suggested by data: =======================================
FullModel = lm(formula = WeightLB ~ Height60IN + factor(DietGroup) + Height60IN:factor(DietGroup), data = DietData)

# examining assumptions and leverage of fit
plot(FullModel)

# looking at ANOVA table
anova(FullModel)

# looking at parameter summary
summary(FullModel)

# building Stan code for same model -- initially wihtout priors

model01_Syntax = "

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
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

group2 = rep(0, nrow(DietData))
group2[which(DietData$DietGroup == 2)] = 1

group3 = rep(0, nrow(DietData))
group3[which(DietData$DietGroup == 3)] = 1

heightXgroup2 = DietData$Height60IN*group2
heightXgroup3 = DietData$Height60IN*group3

# building Stan data into R list
mode01_StanData = list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)


model01_Stan = cmdstan_model(stan_file = write_stan_file(model01_Syntax))

model01_Samples = model01_Stan$sample(
  data = mode01_StanData,
  seed = 19092022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10000,
  iter_sampling = 10000
)

# assess convergence: summary of all parameters
model01_Samples$summary()

# maximum R-hat
max(model01_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model01_Samples$draws())

# posterior histograms
mcmc_hist(model01_Samples$draws())

# posterior densities
mcmc_dens(model01_Samples$draws())

# next: compare model results with results from lm()

stanSummary01 = model01_Samples$summary()
lsSummary01 = summary(FullModel)

# comparison of fixed effects
cbind(lsSummary01$coefficients[,1:2], stanSummary01[2:7,c(2,4)])

# What do you notice being different?


# comparison of residual standard deviation
cbind(lsSummary01$sigma, stanSummary01[8,c(2:4)])

# What do you notice being different?

mcmc_dens(model01_Samples$draws(variables = "sigma"))

# calculating mode of posterior for sigma
mlv(model01_Samples$draws("sigma"), method = "meanshift")


# building model to mirror ls model
model02_Syntax = "

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
}


model {
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, 7.949437);
}

"

model02_Stan = cmdstan_model(stan_file = write_stan_file(model02_Syntax))

model02_Samples = model02_Stan$sample(
  data = mode01_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 100000
)

# assess convergence: summary of all parameters
model02_Samples$summary()

# maximum R-hat
max(model02_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model02_Samples$draws())

# posterior histograms
mcmc_hist(model02_Samples$draws())

# posterior densities
mcmc_dens(model02_Samples$draws())

# comparison of results
stanSummary02 = model02_Samples$summary()
lsSummary02 = summary(FullModel)

# comparison of fixed effects
cbind(lsSummary02$coefficients[,1:2], stanSummary02[2:7,c(2,4)])

# What do you notice being different?

# Investigating Priors

# Prior for sigma: Starting with the exponential distribution
# https://en.wikipedia.org/wiki/Exponential_distribution

?rexp # show help for exponential distribution

# need: value specified for hyperparameter lambda

lambda = c(.1, .5, 1, 5, 10)
sigma = seq(0,1000, .01)
y = cbind(
  dexp(x = sigma, rate = lambda[1]),
  dexp(x = sigma, rate = lambda[2]),
  dexp(x = sigma, rate = lambda[3]),
  dexp(x = sigma, rate = lambda[4]),
  dexp(x = sigma, rate = lambda[5])
)

x = cbind(sigma, sigma, sigma, sigma, sigma)

matplot(x = x, y = y, type = "l", lty = 1:5, col=1:5, lwd = 2)
legend(x = 50, y = 5, legend = paste0(lambda), lty = 1:5, col=1:5, lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col=1:5, xlim=c(0,100), lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col=1:5, xlim=c(0,10), lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col=1:5, xlim=c(0,4), lwd = 2)

# adding prior variance to sigma
model03_Syntax = "

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
  sigma ~ exponential(10); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

model03_Stan = cmdstan_model(stan_file = write_stan_file(model03_Syntax))

model03_Samples = model03_Stan$sample(
  data = mode01_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
model03_Samples$summary()

# maximum R-hat
max(model03_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model03_Samples$draws())

# posterior histograms
mcmc_hist(model03_Samples$draws())

# posterior densities
mcmc_dens(model03_Samples$draws())

# comparison of results
stanSummary03 = model03_Samples$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8,2:4], stanSummary03[2:8,2:4])


cbind(lsSummary01$coefficients[,1:2], stanSummary03[2:7,c(2,4)])

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
  beta0 ~ normal(0,1);
  betaHeight ~ normal(0,1);
  betaGroup2 ~ normal(0,1);
  betaGroup3 ~ normal(0,1);
  betaHxG2 ~ normal(0,1);
  betaHxG3 ~ normal(0,1);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

model04_Stan = cmdstan_model(stan_file = write_stan_file(model04_Syntax))

model04_Samples = model04_Stan$sample(
  data = mode01_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
model04_Samples$summary()

# maximum R-hat
max(model04_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model04_Samples$draws())

# posterior histograms
mcmc_hist(model04_Samples$draws())

# posterior densities
mcmc_dens(model04_Samples$draws())

# comparison of results
stanSummary04 = model04_Samples$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8,2:4], stanSummary04[2:8,2:4])

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

# visualize posterior timeseries
mcmc_trace(model05_Samples$draws())

# posterior histograms
mcmc_hist(model05_Samples$draws())

# posterior densities
mcmc_dens(model05_Samples$draws())

# comparison of results
stanSummary05 = model05_Samples$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8,2:4], stanSummary05[2:8,2:4])

# generated quantities
# adding prior to betas
model06_Syntax = "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;     // prior mean vector for coefficients
  matrix[P, P] covBeta;   // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
  
  int<lower=0> nContrasts; 
  matrix[nContrasts,P] contrast;   // contrast matrix for additional effects
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
  vector[nContrasts] heightSlopeG2;
  real rss;
  real totalrss;
    
  heightSlopeG2 = contrast*beta;
  
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

nContrasts = 1
contrast = matrix(data = 0, nrow = 1, ncol = P)
contrast[1,2] = contrast[1,4] = 1

# build Stan data from model matrix
model06_data = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate,
  contrast = contrast,
  nContrasts = nContrasts
)

model06_Stan = cmdstan_model(stan_file = write_stan_file(model06_Syntax))

model06_Samples = model06_Stan$sample(
  data = model06_data,
  seed = 23092022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
model06_Samples$summary()

# assess convergence: summary of all parameters
max(model06_Samples$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(model06_Samples$draws())

# posterior histograms
mcmc_hist(model06_Samples$draws("R2"))

# posterior densities
mcmc_dens(model06_Samples$draws())

# 
# b0 + b1*H + b2*G2 + b3* G3 + b4*H*G2 + b5*H*G3
# 
# When G2=1
# b0 + b1*H + b2 + B4*H  = b0+b2 + b1+b4*H

# which models fit the best?
# how can we tell?

# next week:
#   model comparison with WAIC
#   posterior predictive model checks
#   combinations of parameters


# Wednesday's class
# 1. finish example (culminating with matrices)
    # a. Add effects for variances
# 2. add generated quantities for example for R^2 and for new effects
#

