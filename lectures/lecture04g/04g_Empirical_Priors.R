# Package installation ======================================================================
needed_packages = c("ggplot2", "cmdstanr", "HDInterval", "bayesplot", "loo")
for(i in 1:length(needed_packages)){
  haspackage = require(needed_packages[i], character.only = TRUE)
  if(haspackage == FALSE){
    install.packages(needed_packages[i])
  }
  library(needed_packages[i], character.only = TRUE)
}
# set number of cores to 4 for this analysis
options(mc.cores = 4)

# Import data ===============================================================================

conspiracyData = read.csv("conspiracies.csv")
conspiracyItems = conspiracyData[,1:10]

# Import data ===============================================================================

conspiracyData = read.csv("conspiracies.csv")
conspiracyItems = conspiracyData[,1:10]

# CFA Model Syntax ==========================================================================

modelCFA_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  vector[nItems] meanMu;
  matrix[nItems, nItems] covMu;      // prior covariance matrix for coefficients
  
  vector[nItems] meanLambda;         // prior mean vector for coefficients
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for coefficients
  
  vector[nItems] psiRate;            // prior rate parameter for unique standard deviations
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
  vector<lower=0>[nItems] psi;       // the unique standard deviations (one for each item)   
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  psi ~ exponential(psiRate);                   // Prior for unique standard deviations
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"

modelCFA_stan = cmdstan_model(stan_file = write_stan_file(modelCFA_syntax))

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item intercept hyperparameters
muMeanHyperParameter = 0
muMeanVecHP = rep(muMeanHyperParameter, nItems)

muVarianceHyperParameter = 1000
muCovarianceMatrixHP = diag(x = muVarianceHyperParameter, nrow = nItems)

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)

# unique standard deviation hyperparameters
psiRateHyperParameter = .01
psiRateVecHP = rep(psiRateHyperParameter, nItems)

modelCFA_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanMu = muMeanVecHP,
  covMu = muCovarianceMatrixHP,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  psiRate = psiRateVecHP
)

modelCFA_samples = modelCFA_stan$sample(
  data = modelCFA_data,
  seed = 19112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelCFA_samples$summary(variables = c("mu", "lambda", "psi")) ,n=Inf)

# Empirical Prior on Item Parameters ===============================================

modelCFA2_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  real meanLambdaMean;
  real<lower=0> meanLambdaSD;
  real<lower=0> sdLambdaRate;
  
  real meanMuMean;
  real<lower=0> meanMuSD;
  real<lower=0> sdMuRate;
  
  real<lower=0> ratePsiRate;
}

parameters {
  vector[nObs] theta;                
  vector[nItems] mu;                 
  vector[nItems] lambda;            
  vector<lower=0>[nItems] psi;      
  real meanLambda;
  real<lower=0> sdLambda;
  
  real meanMu;
  real<lower=0> sdMu;
  real<lower=0> psiRate;
}

model {
  
  meanLambda ~ normal(meanLambdaMean, meanLambdaSD);
  sdLambda ~ exponential(sdLambdaRate);
  lambda ~ normal(meanLambda, sdLambda);
  
  meanMu ~ normal(meanMuMean, meanMuSD);
  sdMu ~ exponential(sdMuRate);
  mu ~ normal(meanMu, sdMu); 
  
  psiRate ~ exponential(ratePsiRate);
  psi ~ exponential(psiRate);            
  
  theta ~ normal(0, 1);                  
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"

modelCFA2_stan = cmdstan_model(stan_file = write_stan_file(modelCFA2_syntax))


modelCFA2_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanLambdaMean = 0,
  meanLambdaSD = 1,
  sdLambdaRate = .1,
  meanMuMean = 0,
  meanMuSD = 1,
  sdMuRate = .1,
  ratePsiRate = .1
)


modelCFA2_samples = modelCFA2_stan$sample(
  data = modelCFA2_data,
  seed = 191120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA2_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(
  modelCFA2_samples$summary(
    variables = c("mu", "meanMu", "sdMu", "lambda", "meanLambda", "sdLambda", "psi", "psiRate")
    ), 
  n=Inf
)


# comparing lambda estimates: uninformative vs. empirical prior
plot(x = modelCFA_samples$summary(variables = "lambda")$mean,
     y = modelCFA2_samples$summary(variables = "lambda")$mean, 
     ylab = "Empirical Prior", xlab = "Uninformative Prior", main = "Comparing EAPs for Lambda")

hist(modelCFA_samples$summary(variables = "lambda")$mean - modelCFA2_samples$summary(variables = "lambda")$mean,
     xlab = "Lambda EAP Difference", main = "Uninformative Lambda Prior EAP(lambda) - Empirical Lambda Prior EAP(lambda)")

plot(x = modelCFA_samples$summary(variables = "lambda")$sd,
     y = modelCFA2_samples$summary(variables = "lambda")$sd, 
     ylab = "Empirical Prior", xlab = "Uninformative Prior", main = "Comparing Posterior SDs for Lambda")

hist(modelCFA_samples$summary(variables = "lambda")$sd - modelCFA2_samples$summary(variables = "lambda")$sd,
     xlab = "Lambda SD Difference", main = "Uninformative Lambda Prior SD(lambda) - Empirical Lambda Prior SD(lambda)")

plot(x = modelCFA_samples$summary(variables = "theta")$mean,
     y = modelCFA2_samples$summary(variables = "theta")$mean, 
     ylab = "Theta Empirical Prior", xlab = "Theta Uninformative Prior", main = "Comparing EAP Estimates for Theta")

hist(modelCFA_samples$summary(variables = "theta")$mean - modelCFA2_samples$summary(variables = "theta")$mean,
     xlab = "Theta EAP Difference", main = "Uninformative Theta Prior EAP(theta) - Empirical Thets Prior EAP(theta)")

plot(x = modelCFA_samples$summary(variables = "theta")$sd,
     y = modelCFA2_samples$summary(variables = "theta")$sd, 
     ylab = "Theta Empirical Prior", xlab = "Theta Uninformative Prior", main = "Comparing SDs for Theta")

hist(modelCFA_samples$summary(variables = "theta")$sd - modelCFA2_samples$summary(variables = "theta")$sd,
     xlab = "Theta SD Difference", main = "Uninformative Theta Prior SD(theta) - Empirical Theta Prior SD(theta)")


# empirical prior for theta ===========================================================

modelCFA3_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  real meanLambdaMean;
  real<lower=0> meanLambdaSD;
  real<lower=0> sdLambdaRate;
  
  real meanMuMean;
  real<lower=0> meanMuSD;
  real<lower=0> sdMuRate;
  
  real<lower=0> ratePsiRate;
  
  real meanThetaMean;
  real<lower=0> meanThetaSD;
  real<lower=0> sdThetaRate;
}

parameters {
  vector[nObs] theta;                
  real meanTheta;
  real<lower=0> sdTheta;
    
  vector[nItems] lambda;
  real meanLambda;
  real<lower=0> sdLambda;
  
  vector[nItems] mu;
  real meanMu;
  real<lower=0> sdMu;
  
  vector<lower=0>[nItems] psi;
  real<lower=0> psiRate;
}

model {
  
  meanLambda ~ normal(meanLambdaMean, meanLambdaSD);
  sdLambda ~ exponential(sdLambdaRate);
  lambda ~ normal(meanLambda, sdLambda);
  
  meanMu ~ normal(meanMuMean, meanMuSD);
  sdMu ~ exponential(sdMuRate);
  mu ~ normal(meanMu, sdMu); 
  
  psiRate ~ exponential(ratePsiRate);
  psi ~ exponential(psiRate);            
  
  meanTheta ~ normal(meanThetaMean, meanThetaSD);
  sdTheta ~ exponential(sdThetaRate);
  theta ~ normal(meanTheta, sdTheta);                  
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"
modelCFA3_stan = cmdstan_model(stan_file = write_stan_file(modelCFA3_syntax))

modelCFA3_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanLambdaMean = 0,
  meanLambdaSD = 1,
  sdLambdaRate = .1,
  meanMuMean = 0,
  meanMuSD = 1,
  sdMuRate = .1,
  ratePsiRate = .1,
  meanThetaMean = 0,
  meanThetaSD = 1,
  sdThetaRate = .1
)


modelCFA3_samples = modelCFA3_stan$sample(
  data = modelCFA3_data,
  seed = 191120223,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA3_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(
  modelCFA3_samples$summary(
    variables = c("meanTheta", "sdTheta", "mu", "meanMu", "sdMu", "lambda", "meanLambda", "sdLambda", "psi", "psiRate")
  ), 
  n=Inf
)

mcmc_trace(modelCFA3_samples$draws(variables = c("meanTheta", "sdTheta")))

# model4: empirical theta prior, fixed prior on item parameters
modelCFA4_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  vector[nItems] meanMu;
  matrix[nItems, nItems] covMu;      // prior covariance matrix for coefficients
  
  vector[nItems] meanLambda;         // prior mean vector for coefficients
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for coefficients
  
  vector[nItems] psiRate;            // prior rate parameter for unique standard deviations
  
  real meanThetaMean;
  real<lower=0> meanThetaSD;
  real<lower=0> sdThetaRate;
  
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  real meanTheta;
  real<lower=0> sdTheta;
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
  vector<lower=0>[nItems] psi;       // the unique standard deviations (one for each item)   
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  psi ~ exponential(psiRate);                   // Prior for unique standard deviations
  
  meanTheta ~ normal(meanThetaMean, meanThetaSD);
  sdTheta ~ exponential(sdThetaRate);
  theta ~ normal(meanTheta, sdTheta);                  
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"

modelCFA4_stan = cmdstan_model(stan_file = write_stan_file(modelCFA4_syntax))


# item intercept hyperparameters
muMeanHyperParameter = 0
muMeanVecHP = rep(muMeanHyperParameter, nItems)

muVarianceHyperParameter = 1
muCovarianceMatrixHP = diag(x = muVarianceHyperParameter, nrow = nItems)

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 1
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)

# unique standard deviation hyperparameters
psiRateHyperParameter = .01
psiRateVecHP = rep(psiRateHyperParameter, nItems)


modelCFA4_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanMu = muMeanVecHP,
  covMu = muCovarianceMatrixHP,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  psiRate = psiRateVecHP,
  meanThetaMean = 0,
  meanThetaSD = 1,
  sdThetaRate = 1
)

modelCFA4_samples = modelCFA4_stan$sample(
  data = modelCFA4_data,
  seed = 191120224,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA4_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelCFA4_samples$summary(variables = c("meanTheta", "sdTheta", "mu", "lambda", "psi")) ,n=Inf)

save.image(file = "lecture04g.RData")


