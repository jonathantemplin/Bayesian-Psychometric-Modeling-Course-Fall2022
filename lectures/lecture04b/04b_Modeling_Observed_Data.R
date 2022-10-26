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

# investigating item parameters
itemNumber = 3

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


# investigating latent variables

#results
print(modelCFA_samples$summary(variables = c("theta")) ,n=Inf)

# EAP distribution
hist(modelCFA_samples$summary(variables = c("theta"))$mean, main="EAP Estimates of Theta", 
     xlab = expression(theta))

plot(density(modelCFA_samples$summary(variables = c("theta"))$mean), main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Density of All Posterior Draws
allThetas = modelCFA_samples$draws(variables = c("theta"), format="draws_matrix")
allThetasVec = c(allThetas)
hist(allThetasVec)


# plotting two theta distributions side-by-side
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelCFA_samples$draws(variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
                     sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

## comparing EAP estimates with posterior SDs

plot(y = modelCFA_samples$summary(variables = c("theta"))$sd, x = modelCFA_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)")

## comparing EAP estimates with sum scores
plot(x = rowSums(conspiracyItems), y = modelCFA_samples$summary(variables = c("theta"))$mean,
     xlab = "Sum Score", ylab = expression(theta))

# Fails

modelCFA_samplesFail = modelCFA_stan$sample(
  data = modelCFA_data,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

# checking convergence
max(modelCFA_samplesFail$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelCFA_samplesFail$summary(variables = c("mu", "lambda", "psi")) ,n=Inf)

# plotting trace
mcmc_trace(modelCFA_samplesFail$draws(variables = "lambda"))

# plotting densities
mcmc_dens(modelCFA_samplesFail$draws(variables = "lambda"))

modelCFAFixed_syntax = "

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
  vector<lower=0>[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
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

modelCFAFixed_stan = cmdstan_model(stan_file = write_stan_file(modelCFAFixed_syntax))

modelCFA_samplesfixed = modelCFAFixed_stan$sample(
  data = modelCFA_data,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)
max(modelCFA_samplesfixed$summary()$rhat, na.rm = TRUE)
print(modelCFA_samplesfixed$summary(variables = c("mu", "lambda", "psi")) ,n=Inf)

save.image(file = "lecture04b.RData")


