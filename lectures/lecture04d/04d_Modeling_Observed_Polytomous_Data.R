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
  seed = 09102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

# checking convergence
max(modelCFA_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelCFA_samples$summary(variables = c("mu", "lambda", "psi")) ,n=Inf)

# investigating item parameters ================================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
labelPsi = paste0("psi[", itemNumber, "]")
itemParameters = modelCFA_samples$draws(variables = c(labelMu, labelLambda, labelPsi), format = "draws_matrix")
itemSummary = modelCFA_samples$summary(variables = c(labelMu, labelLambda, labelPsi))

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values

# drawing item characteristic curves for item
y = as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta
plot(x = theta, y = y, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(-2,8), xlab = expression(theta), ylab=paste("Item", itemNumber,"Predicted Value"))
for (draw in 2:nrow(itemParameters)){
  y = as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta
  lines(x = theta, y = y)
}

# drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)

# drawing EAP line
y = itemSummary$mean[which(itemSummary$variable==labelMu)] + 
  itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta
lines(x = theta, y = y, lwd = 5, lty=3, col=2)

# legend
legend(x = -3, y = 7, legend = c("Posterior Draw", "Item Limits", "EAP"), col = c(1,4,2), lty = c(1,2,3), lwd=5)



# Binomial Model Syntax (slope/intercept form ) ==================================================

# note: data must start at zero
conspiracyItemsBinomial = conspiracyItems
for (item in 1:ncol(conspiracyItemsBinomial)){
  conspiracyItemsBinomial[, item] = conspiracyItemsBinomial[, item] - 1
}

# check first item
table(conspiracyItemsBinomial[,1])

# determine maximum value for each item
maxItem = apply(X = conspiracyItemsBinomial,
                MARGIN = 2, 
                FUN = max)

modelBinomial_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  array[nItems] int<lower=0> maxItem;
  
  array[nItems, nObs] int<lower=0>  Y; // item responses in an array

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
    Y[item] ~ binomial(maxItem[item], inv_logit(mu[item] + lambda[item]*theta));
  }
  
}

"

modelBinomial_stan = cmdstan_model(stan_file = write_stan_file(modelBinomial_syntax))

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


modelBinomial_data = list(
  nObs = nObs,
  nItems = nItems,
  maxItem = maxItem,
  Y = t(conspiracyItemsBinomial), 
  meanMu = muMeanVecHP,
  covMu = muCovarianceMatrixHP,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP
)

modelBinomial_samples = modelBinomial_stan$sample(
  data = modelBinomial_data,
  seed = 12112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelBinomial_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelBinomial_samples$summary(variables = c("mu", "lambda")) ,n=Inf)

# investigating option characteristic curves ===================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
itemParameters = modelBinomial_samples$draws(variables = c(labelMu, labelLambda), format = "draws_matrix")
itemSummary = modelBinomial_samples$summary(variables = c(labelMu, labelLambda))

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values
y = matrix(data = 0, nrow = length(theta), ncol=5)

thetaMat = NULL
prob = exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
             itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta)/
  (1+exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
           itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta))

option = 1
for (option in 1:5){
  thetaMat = cbind(thetaMat, theta)
  y[,option] = dbinom(x = option-1, size=4, prob=prob)
}

matplot(x = thetaMat, y = y, type="l", xlab=expression(theta), ylab="P(Y |theta)", 
        main=paste0("Option Characteristic Curves for Item ", itemNumber), lwd=3)

legend(x = -3, y = .8, legend = paste("Option", 1:5), lty = 1:5, col=1:5, lwd=3)



# investigating item parameters ================================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
itemParameters = modelBinomial_samples$draws(variables = c(labelMu, labelLambda), format = "draws_matrix")
itemSummary = modelBinomial_samples$summary(variables = c(labelMu, labelLambda))

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values

# drawing item characteristic curves for item
y = 4*exp(as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta)/
  (1+exp(as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta)) +1
plot(x = theta, y = y, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(0,6), xlab = expression(theta), ylab=paste("Item", itemNumber,"Expected Value"))
for (draw in 2:nrow(itemParameters)){
  y =4*exp(as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta)/
    (1+exp(as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta)) +1 
  lines(x = theta, y = y)
}

# drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)

# drawing EAP line
y = 4*exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
  itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta)/
  (1+exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
           itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta)) +1
lines(x = theta, y = y, lwd = 5, lty=3, col=2)

# legend
legend(x = -3, y = 4, legend = c("Posterior Draw", "Item Limits", "EAP"), col = c(1,4,2), lty = c(1,2,3), lwd=5)


# EAP Estimates of Latent Variables
hist(modelBinomial_samples$summary(variables = c("theta"))$mean, main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Comparing Two Posterior Distributions
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelBinomial_samples$draws(variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
                     sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

# Comparing EAP Estimates with Posterior SDs
plot(y = modelBinomial_samples$summary(variables = c("theta"))$sd, 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItemsBinomial), 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Binomial vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Binomial")

# Ordered Logit (Multinomial/categorical distribution) Model Syntax =======================



modelOrderedLogit_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory; 
  
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  array[nItems] vector[maxCategory-1] meanThr;   // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;      // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);             // Prior for item thresholds
    Y[item] ~ ordered_logistic(lambda[item]*theta, thr[item]);
  }
  
}

generated quantities{
  array[nItems] vector[maxCategory-1] mu;
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
}

"

modelOrderedLogit_stan = cmdstan_model(stan_file = write_stan_file(modelOrderedLogit_syntax))


# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item threshold hyperparameters
thrMeanHyperParameter = 0
thrMeanVecHP = rep(thrMeanHyperParameter, maxCategory-1)
thrMeanMatrix = NULL
for (item in 1:nItems){
  thrMeanMatrix = rbind(thrMeanMatrix, thrMeanVecHP)
}

thrVarianceHyperParameter = 1000
thrCovarianceMatrixHP = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
thrCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  thrCovArray[item, , ] = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)


modelOrderedLogit_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  maxItem = maxItem,
  Y = t(conspiracyItems), 
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP
)

modelOrderedLogit_samples = modelOrderedLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelOrderedLogit_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelOrderedLogit_samples$summary(variables = c("lambda", "mu")) ,n=Inf)


## investigating option characteristic curves ===================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, ",", 1:4, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
muParams = modelOrderedLogit_samples$summary(variables = labelMu)
lambdaParams = modelOrderedLogit_samples$summary(variables = labelLambda)

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values
y = NULL
thetaMat = NULL
expectedValue = 0

option = 1
for (option in 1:5){
  if (option==1){
    prob = 1 - exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
                 lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
      (1+exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
               lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))
  } else if (option == 5){
    
    prob = (exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                  lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
              (1+exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                       lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)))
  } else {
    prob = (exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                  lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
              (1+exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                       lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))) -
      exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
            lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
      (1+exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
               lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))
  }
  
  thetaMat = cbind(thetaMat, theta)
  expectedValue = expectedValue + prob*option
  y = cbind(y, prob)
}

matplot(x = thetaMat, y = y, type="l", xlab=expression(theta), ylab="P(Y |theta)", 
        main=paste0("Option Characteristic Curves for Item ", itemNumber), lwd=3)

legend(x = -3, y = .8, legend = paste("Option", 1:5), lty = 1:5, col=1:5, lwd=3)

## plot of EAP of expected value per item ======================================================
plot(x = theta, y = expectedValue, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(0,6), xlab = expression(theta), ylab=paste("Item", itemNumber,"Expected Value"), lwd = 5, lty=3, col=2)

# drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)

# EAP Estimates of Latent Variables
hist(modelOrderedLogit_samples$summary(variables = c("theta"))$mean, 
     main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Comparing Two Posterior Distributions
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelOrderedLogit_samples$draws(
  variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(
  observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
  sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

# Comparing EAP Estimates with Posterior SDs

plot(y = modelOrderedLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItems), 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Ordered Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Ordered Logit")

# Comparing Theta SDs: Ordered Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Normal", xlab = "Ordered Logit", main="Posterior SDs")

# Which is bigger?
hist(modelCFA_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(normal) - SD(ordered)")

# Comparing Thetas: Ordered Logit vs Binomial:
plot(y = modelBinomial_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Binomial", xlab = "Ordered Logit")

# Comparing Theta SDs: Ordered Logit vs Binomial:
plot(y = modelBinomial_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Binomial", xlab = "Ordered Logit", main="Posterior SDs")

# Which is bigger?
hist(modelBinomial_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(binomial) - SD(ordered)")



# Categorical Logit (Multinomial/categorical distribution) Model Syntax =======================
# Also known as: Nominal Response Model (in IRT literature) 


modelCategoricalLogit_syntax = "
data {
  int maxCategory;
  int nObs;
  int nItems;
  
  array[nItems, nObs] int Y; 
  
  array[nItems] vector[maxCategory-1] meanMu;   // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covMu;      // prior covariance matrix for intercept parameters
  
  array[nItems] vector[maxCategory-1] meanLambda;       // prior mean vector for discrimination parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covLambda;  // prior covariance matrix for discrimination parameters
  
}

parameters {
  array[nItems] vector[maxCategory - 1] initMu;
  array[nItems] vector[maxCategory - 1] initLambda;
  vector[nObs] theta;                // the latent variables (one for each person)
}

transformed parameters {
  array[nItems] vector[maxCategory] mu;
  array[nItems] vector[maxCategory] lambda;
  
  for (item in 1:nItems){
    mu[item, 2:maxCategory] = initMu[item, 1:(maxCategory-1)];
    mu[item, 1] = 0.0; // setting one category's intercept to zero
    
    lambda[item, 2:maxCategory] = initLambda[item, 1:(maxCategory-1)];
    lambda[item, 1] = 0.0; // setting one category's lambda to zero
    
  }
}

model {
  
  vector[maxCategory] probVec;
  
  theta ~ normal(0,1);
  
  for (item in 1:nItems){
    for (category in 1:(maxCategory-1)){
      initMu[item, category] ~ normal(meanMu[item, category], covMu[item, category, category]);  // Prior for item intercepts
      initLambda[item, category] ~ normal(meanLambda[item, category], covLambda[item, category, category]);  // Prior for item loadings
    }
  }
    
  for (obs in 1:nObs) {
    for (item in 1:nItems){
      for (category in 1:maxCategory){
        probVec[category] = mu[item, category] + lambda[item, category]*theta[obs];     
      }
      Y[item, obs] ~ categorical_logit(probVec);
    }  
  }
}
"


modelCategoricalLogit_stan = cmdstan_model(stan_file = write_stan_file(modelCategoricalLogit_syntax))

# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item threshold hyperparameters
muMeanHyperParameter = 0
muMeanVecHP = rep(muMeanHyperParameter, maxCategory-1)
muMeanMatrix = NULL
for (item in 1:nItems){
  muMeanMatrix = rbind(muMeanMatrix, muMeanVecHP)
}

muVarianceHyperParameter = 1
muCovarianceMatrixHP = diag(x = muVarianceHyperParameter, nrow = maxCategory-1)
muCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  muCovArray[item, , ] = diag(x = muVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, maxCategory-1)
lambdaMeanMatrix = NULL
for (item in 1:nItems){
  lambdaMeanMatrix = rbind(lambdaMeanMatrix, lambdaMeanVecHP)
}

lambdaVarianceHyperParameter = 1
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = maxCategory-1)
lambdaCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  lambdaCovArray[item, , ] = diag(x = lambdaVarianceHyperParameter, nrow = maxCategory-1)
}


modelOrderedLogit_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  maxItem = maxItem,
  Y = t(conspiracyItems), 
  meanMu = muMeanMatrix,
  covMu = muCovArray,
  meanLambda = lambdaMeanMatrix,
  covLambda = lambdaCovArray
)

# for checking initial values:
modelCategoricalLogit_samples = modelCategoricalLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 0,
  iter_sampling = 5,
  init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=-1, sd=.1)), 
  adapt_engaged = FALSE
)
modelCategoricalLogit_samples$draws(variables = "initLambda")

modelCategoricalLogit_samples = modelCategoricalLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=1, sd=.1))
)

# checking convergence
max(modelCategoricalLogit_samples$summary()$rhat, na.rm = TRUE)

print(modelCategoricalLogit_samples$summary(variables = c("mu", "lambda")), n=Inf)

## investigating option characteristic curves ===================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, ",", 1:5, "]")
muParams = modelCategoricalLogit_samples$summary(variables = labelMu)

labelLambda = paste0("lambda[", itemNumber, ",", 1:5, "]")
lambdaParams = modelCategoricalLogit_samples$summary(variables = labelLambda)

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values
thetaMat = NULL

logit=NULL
prob=NULL
probsum = 0
option = 1
for (option in 1:5){
  logit = cbind(logit, muParams$mean[option] + lambdaParams$mean[option]*theta)
  prob = cbind(prob, exp(logit[,option]))
  probsum = probsum+ exp(logit[,option])
}

for (option in 1:5){
  thetaMat = cbind(thetaMat, theta)
  prob[,option] = prob[,option]/probsum
}

matplot(x = thetaMat, y = prob, type="l", xlab=expression(theta), ylab="P(Y |theta)", 
        main=paste0("Option Characteristic Curves for Item ", itemNumber), lwd=3)

legend(x = -3, y = .8, legend = paste("Option", 1:5), lty = 1:5, col=1:5, lwd=3)


# Comparing EAP Estimates with Posterior SDs

plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItems), 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Categorical Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Categorical Logit")

# Comparing Theta SDs: Categorical Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$sd, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Normal", xlab = "Categorical Logit", main="Posterior SDs")

# Which is bigger?
hist(modelCFA_samples$summary(variables = c("theta"))$sd-
       modelCategoricalLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(normal) - SD(categorical)")

# Comparing Thetas: Categorical Logit vs Ordinal:
plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "NRM", xlab = "GRM")

# Comparing Theta SDs: Ordered Logit vs Binomial:
plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "NRM", xlab = "GRM", main="Posterior SDs")

# Which is bigger?
hist(modelCategoricalLogit_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(NRM) - SD(GRM)")

