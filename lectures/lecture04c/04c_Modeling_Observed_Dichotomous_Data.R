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

# Converting to dichotomous responses (only for teaching: don't do to your data)=============
# Here 0 == strongly disagree or disagree; 1 == neither, agree, and strongly disagree =======

conspiracyItemsDichtomous = conspiracyItems
for (var in 1:ncol(conspiracyItems)){
  conspiracyItemsDichtomous[which(conspiracyItemsDichtomous[,var] <=3),var] = 0
  conspiracyItemsDichtomous[which(conspiracyItemsDichtomous[,var] > 3),var] = 1
}

# examining data after transformation
table(conspiracyItemsDichtomous$PolConsp1, conspiracyItems$PolConsp1)

# item means:
apply(X = conspiracyItemsDichtomous, MARGIN = 2, FUN = mean)
  
# Example Likelihood Functions ==============================================================

# for lambda1 
mu1 = -2
theta = rnorm(n = nrow(conspiracyItemsDichtomous), mean = 0, sd = 1)

lambda = seq(-2,2, .01)
logLike = NULL

param=1 # for demonstrating
for (param in 1:length(lambda)){
  
  logit = mu1 + lambda[param]*theta
  prob = exp(logit)/(1+exp(logit))
  bernoulliLL = sum(dbinom(x = conspiracyItemsDichtomous$PolConsp1, size = 1, prob = prob, log = TRUE))
  
  logLike = c(logLike, bernoulliLL)
}

plot(x = lambda, y = logLike, type = "l")

# for theta2
mu = runif(n = ncol(conspiracyItemsDichtomous), min = -2, max = 0)
lambda = runif(n = ncol(conspiracyItemsDichtomous), min = 0, max = 2)

person = 2

theta = seq(-3,3,.01)
logLike = NULL

param=1 # for demonstrating
for (param in 1:length(theta)){
  thetaLL = 0
  for (item in 1:ncol(conspiracyItemsDichtomous)){
    logit = mu[item] + lambda[item]*theta[param]
    prob = exp(logit)/(1+exp(logit))
    thetaLL = thetaLL + dbinom(x = conspiracyItemsDichtomous[person,item], size = 1, prob = prob, log = TRUE)
  }
  
  logLike = c(logLike, thetaLL)
}

plot(x = theta, y = logLike, type = "l")


# IRT Model Syntax (slope/intercept form ) ==================================================


modelIRT_2PL_SI_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in an array

  vector[nItems] meanMu;             // prior mean vector for intercept parameters
  matrix[nItems, nItems] covMu;      // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(mu[item] + lambda[item]*theta);
  }
  
}

"

modelIRT_2PL_SI_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_SI_syntax))

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


modelIRT_2PL_SI_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanMu = muMeanVecHP,
  covMu = muCovarianceMatrixHP,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP
)

modelIRT_2PL_SI_samples = modelIRT_2PL_SI_stan$sample(
  data = modelIRT_2PL_SI_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_SI_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_SI_samples$summary(variables = c("mu", "lambda")) ,n=Inf)

# investigating item parameters ================================================
itemNumber = 5

labelMu = paste0("mu[", itemNumber, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
itemParameters = modelIRT_2PL_SI_samples$draws(variables = c(labelMu, labelLambda), format = "draws_matrix")
itemSummary = modelIRT_2PL_SI_samples$summary(variables = c(labelMu, labelLambda))

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values

# drawing item characteristic curves for item
logit = as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta
y = exp(logit)/(1+exp(logit))
plot(x = theta, y = y, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(0,1), xlab = expression(theta), ylab=paste("Item", itemNumber, "Predicted Value"))

for (draw in 2:nrow(itemParameters)){
  logit = as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta
  y = exp(logit)/(1+exp(logit))
  lines(x = theta, y = y)
}

# drawing EAP line
logit = itemSummary$mean[which(itemSummary$variable==labelMu)] + 
  itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta
y = exp(logit)/(1+exp(logit))
lines(x = theta, y = y, lwd = 5, lty=3, col=2)

# legend
legend(x = -3, y = 1, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,3), lwd=5)

# investigating item parameters
mcmc_trace(modelIRT_2PL_SI_samples$draws(variables = "mu"))
mcmc_dens(modelIRT_2PL_SI_samples$draws(variables = "mu"))

mcmc_trace(modelIRT_2PL_SI_samples$draws(variables = "lambda"))
mcmc_dens(modelIRT_2PL_SI_samples$draws(variables = "lambda"))

# bivariate posterior distributions
itemNum = 1
muLabel = paste0("mu[", itemNum, "]")
lambdaLabel = paste0("lambda[", itemNum, "]")
mcmc_pairs(modelIRT_2PL_SI_samples$draws(), pars = c(muLabel, lambdaLabel))

# investigating the latent variables:
print(modelIRT_2PL_SI_samples$summary(variables = "theta") ,n=Inf)


# EAP Estimates of Latent Variables

hist(modelIRT_2PL_SI_samples$summary(variables = c("theta"))$mean, main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Comparing Two Posterior Distributions
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelIRT_2PL_SI_samples$draws(variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
                     sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

# Comparing EAP Estimates with Posterior SDs

plot(y = modelIRT_2PL_SI_samples$summary(variables = c("theta"))$sd, 
     x = modelIRT_2PL_SI_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItemsDichtomous), x = modelIRT_2PL_SI_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))


# IRT Model Syntax (slope/intercept form with discrimination/difficulty calculated) ===========


modelIRT_2PL_SI2_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in an array

  vector[nItems] meanMu;             // prior mean vector for intercept parameters
  matrix[nItems, nItems] covMu;      // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(mu[item] + lambda[item]*theta);
  }
  
}

generated quantities{
  vector[nItems] a;
  vector[nItems] b;
  
  for (item in 1:nItems){
    a[item] = lambda[item];
    b[item] = -1*mu[item]/lambda[item];
  }
  
}

"

modelIRT_2PL_SI2_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_SI2_syntax))


modelIRT_2PL_SI2_samples = modelIRT_2PL_SI2_stan$sample(
  data = modelIRT_2PL_SI_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_SI2_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_SI2_samples$summary(variables = c("a", "b")) ,n=Inf)

# IRT Model Syntax (discrimination/difficulty form ) ==================================================

modelIRT_2PL_DD_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix

  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;                 // the item intercepts (one for each item)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  a ~ multi_normal(meanA, covA); // Prior for item discrimination/factor loadings
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(a[item]*(theta - b[item]));
  }
  
}

generated quantities{
  vector[nItems] lambda;
  vector[nItems] mu;
  
  lambda = a;
  for (item in 1:nItems){
    mu[item] = -1*a[item]*b[item];
  }
}

"

modelIRT_2PL_DD_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_DD_syntax))

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item intercept hyperparameters
bMeanHyperParameter = 0
bMeanVecHP = rep(bMeanHyperParameter, nItems)

bVarianceHyperParameter = 1000
bCovarianceMatrixHP = diag(x = bVarianceHyperParameter, nrow = nItems)

# item discrimination/factor loading hyperparameters
aMeanHyperParameter = 0
aMeanVecHP = rep(aMeanHyperParameter, nItems)

aVarianceHyperParameter = 1000
aCovarianceMatrixHP = diag(x = aVarianceHyperParameter, nrow = nItems)

modelIRT_2PL_DD_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP
)

modelIRT_2PL_DD_samples = modelIRT_2PL_DD_stan$sample(
  data = modelIRT_2PL_DD_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_DD_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_DD_samples$summary(variables = c("a", "b")) ,n=Inf)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("b"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("b"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Difficulty Parameter EAP Estimates"
)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("a"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("a"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Discrimination Parameters EAP Estimates"
)

# theta results
# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("theta"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("theta"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Theta EAP Estimates"
)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("theta"))$sd,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("theta"))$sd,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Theta SD Estimates"
)


# IRT Auxiliary Statistics ===========================================================

modelIRT_2PL_DD2_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix

  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
  
  int<lower=0> nThetas;        // number of theta values for auxiliary statistics
  vector[nThetas] thetaVals;   // values for auxiliary statistics
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;                 // the item intercepts (one for each item)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  a ~ multi_normal(meanA, covA); // Prior for item discrimination/factor loadings
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(a[item]*(theta - b[item]));
  }
  
}

generated quantities{
  vector[nItems] lambda;
  vector[nItems] mu;
  vector[nThetas] TCC;
  matrix[nThetas, nItems] itemInfo;
  vector[nThetas] testInfo;
  
  for (val in 1:nThetas){
    TCC[val] = 0.0;
    testInfo[val] = -1.0;  // test information must start at -1 to include prior distribution for theta
    for (item in 1:nItems){
      itemInfo[val, item] = 0.0;
    }
  }
  
  lambda = a;
  for (item in 1:nItems){
    mu[item] = -1*a[item]*b[item];
    
    for (val in 1:nThetas){
      // test characteristic curve:
      TCC[val] = TCC[val] + inv_logit(a[item]*(thetaVals[val]-b[item]));
      
      // item information functions:
      itemInfo[val, item] = 
        itemInfo[val, item] + 
          a[item]^2*inv_logit(a[item]*(thetaVals[val]-b[item]))*(1-inv_logit(a[item]*(thetaVals[val]-b[item])));
        
      // test information functions:
      testInfo[val] = testInfo[val] + 
        a[item]^2*inv_logit(a[item]*(thetaVals[val]-b[item]))*(1-inv_logit(a[item]*(thetaVals[val]-b[item])));
    }
  }
  
  
  
}

"

modelIRT_2PL_DD2_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_DD2_syntax))

thetaVals = seq(-3,3,.01)

modelIRT_2PL_DD2_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP,
  nThetas = length(thetaVals),
  thetaVals = thetaVals
)

modelIRT_2PL_DD2_samples = modelIRT_2PL_DD2_stan$sample(
  data = modelIRT_2PL_DD2_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_DD2_samples$summary(variables = c("theta", "a", "b"))$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_DD2_samples$summary(variables = c("TCC")) ,n=Inf)


# TCC Spaghetti Plots
tccSamples = modelIRT_2PL_DD2_samples$draws(variables = "TCC", format = "draws_matrix")
plot(x = thetaVals, 
     y = tccSamples[1,],
     xlab = expression(theta), 
     ylab = "Expected Score", type = "l",
     main = "Test Characteristic Curve", lwd = 2)

for (draw in 1:nrow(tccSamples)){
  lines(x = thetaVals,
        y = tccSamples[draw,])
}

# EAP TCC
lines(x = thetaVals, 
      y = modelIRT_2PL_DD2_samples$summary(variables = c("TCC"))$mean,
      lwd = 2, 
      col=2, 
      lty=3)

legend(x = -3, y = 7, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)

# ICC Spaghetti Plots
item = 1
itemLabel = paste0("Item ", item)
iccSamples = modelIRT_2PL_DD2_samples$draws(variables = "itemInfo", format = "draws_matrix")
iccNames = colnames(iccSamples)
itemSamples = iccSamples[,iccNames[grep(pattern = ",1]", x = iccNames)]]

maxInfo = max(apply(X = itemSamples, MARGIN = 2, FUN = max))

plot(x = thetaVals, 
     y = itemSamples[1,],
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = paste0(itemLabel, " Information Function"), lwd = 2,
     ylim = c(0,maxInfo+.5))

for (draw in 1:nrow(itemSamples)){
  lines(x = thetaVals,
        y = itemSamples[draw,])
}

# EAP TCC
lines(x = thetaVals, 
      y = apply(X = itemSamples, MARGIN=2, FUN=mean),
      lwd = 3, 
      col = 2, 
      lty = 3)

legend(x = -3, y = maxInfo-.5, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)


# TIF Spaghetti Plots
tifSamples = modelIRT_2PL_DD2_samples$draws(variables = "testInfo", format = "draws_matrix")
maxTIF = max(apply(X = tifSamples, MARGIN = 2, FUN = max))

plot(x = thetaVals, 
     y = tifSamples[1,],
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", lwd = 2,
     ylim = c(0,maxTIF))

for (draw in 1:nrow(tifSamples)){
  lines(x = thetaVals,
        y = tifSamples[draw,])
}

# EAP TIF
lines(x = thetaVals, 
      y = apply(X=tifSamples, MARGIN=2, FUN=mean),
      lwd = 3, 
      col = 2, 
      lty = 3)

legend(x = -3, y = maxTIF, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)

# EAP TCC
plot(x = thetaVals, 
     y = apply(X=tifSamples, MARGIN=2, FUN=mean),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", 
     lwd = 2)

# Other IRT Models ===========================================================

# 1PL Model:
modelIRT_1PL_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(inv_logit(theta - b[item]));
  }
  
}

"

modelIRT_1PL_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_1PL_syntax))

modelIRT_1PL_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP
)

modelIRT_1PL_samples = modelIRT_1PL_stan$sample(
  data = modelIRT_1PL_data,
  seed = 021120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000
)

# checking convergence
max(modelIRT_1PL_samples$summary(variables = c("theta", "b"))$rhat, na.rm = TRUE)

modelIRT_1PL_samples$summary(variables = c("b"))

# 3PL Model:
modelIRT_3PL_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
  vector<lower=0, upper=1>[nItems] c;
}

model {
  a ~ multi_normal(meanA, covA);             // Prior for item intercepts
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  c ~ beta(1,1);                              // Simple prior for c parameter
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(c[item] + (1-c[item])*inv_logit(a[item]*(theta - b[item])));
  }
  
}

"

modelIRT_3PL_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_3PL_syntax))

modelIRT_3PL_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP
)

modelIRT_3PL_samples = modelIRT_3PL_stan$sample(
  data = modelIRT_3PL_data,
  seed = 021120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_3PL_samples$summary(variables = c("theta", "b", "a", "c"))$rhat, na.rm = TRUE)

print(modelIRT_3PL_samples$summary(variables = c("a", "b", "c")), n=Inf)

# Two-Parameter Normal Ogive Model:
modelIRT_2PNO_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  a ~ multi_normal(meanA, covA);             // Prior for item intercepts
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(Phi(a[item]*(theta - b[item])));
  }
  
}

"

modelIRT_2PNO_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PNO_syntax))

modelIRT_2PNO_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = diag(nItems),
  meanA = aMeanVecHP,
  covA = diag(nItems) # changing prior covariance to help with convergence
)

modelIRT_2PNO_samples = modelIRT_2PNO_stan$sample(
  data = modelIRT_2PNO_data,
  seed = 0211202223,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=3, sd=.05))
)

# checking convergence -- not great!
max(modelIRT_2PNO_samples$summary(variables = c("theta", "b", "a"))$rhat, na.rm = TRUE)

print(modelIRT_2PNO_samples$summary(variables = c("a", "b")), n=Inf)


save.image("lecture04c.RData")


