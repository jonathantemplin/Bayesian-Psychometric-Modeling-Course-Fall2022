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

# Build a Q-Matrix ===========================================================================

Qmatrix = matrix(data = 0, nrow = ncol(conspiracyItems), ncol = 2)
colnames(Qmatrix) = c("Gov", "NonGov")
rownames(Qmatrix) = paste0("item", 1:ncol(conspiracyItems))
Qmatrix[1,2] = 1
Qmatrix[2,1] = 1
Qmatrix[3,2] = 1
Qmatrix[4,2] = 1
Qmatrix[5,1] = 1
Qmatrix[6,2] = 1
Qmatrix[7,1] = 1
Qmatrix[8,1] = 1
Qmatrix[9,1] = 1
Qmatrix[10,2] = 1

Qmatrix

# Ordered Logit (Multinomial/categorical distribution) Model Syntax =======================

modelOrderedLogit_syntax = "
data {
  
  // data specifications  =============================================================
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory;       // number of categories for each item
  
  // input data  =============================================================
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  // loading specifications  =============================================================
  int<lower=1> nFactors;                                       // number of loadings in the model
  array[nItems, nFactors] int<lower=0, upper=1> Qmatrix;
  
  // prior specifications =============================================================
  array[nItems] vector[maxCategory-1] meanThr;                // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;  // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
  
  vector[nFactors] meanTheta;
}

transformed data{
  int<lower=0> nLoadings = 0;                                      // number of loadings in model
  
  for (factor in 1:nFactors){
    nLoadings = nLoadings + sum(Qmatrix[1:nItems, factor]);
  }

  array[nLoadings, 2] int loadingLocation;                     // the row/column positions of each loading
  int loadingNum=1;
  
  for (item in 1:nItems){
    for (factor in 1:nFactors){
      if (Qmatrix[item, factor] == 1){
        loadingLocation[loadingNum, 1] = item;
        loadingLocation[loadingNum, 2] = factor;
        loadingNum = loadingNum + 1;
      }
    }
  }


}

parameters {
  array[nObs] vector[nFactors] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nLoadings] lambda;             // the factor loadings/item discriminations (one for each item)
  cholesky_factor_corr[nFactors] thetaCorrL;
}

transformed parameters{
  matrix[nItems, nFactors] lambdaMatrix = rep_matrix(0.0, nItems, nFactors);
  matrix[nObs, nFactors] thetaMatrix;
  
  // build matrix for lambdas to multiply theta matrix
  for (loading in 1:nLoadings){
    lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = lambda[loading];
  }
  
  for (factor in 1:nFactors){
    thetaMatrix[,factor] = to_vector(theta[,factor]);
  }
  
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); 
  thetaCorrL ~ lkj_corr_cholesky(1.0);
  theta ~ multi_normal_cholesky(meanTheta, thetaCorrL);    
  
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);            
    Y[item] ~ ordered_logistic(thetaMatrix*lambdaMatrix[item,1:nFactors]', thr[item]);
  }
  
  
}

generated quantities{ 
  array[nItems] vector[maxCategory-1] mu;
  corr_matrix[nFactors] thetaCorr;
   
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
  
  
  thetaCorr = multiply_lower_tri_self_transpose(thetaCorrL);
  
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

thrVarianceHyperParameter = 10
thrCovarianceMatrixHP = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
thrCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  thrCovArray[item, , ] = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 10
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)

thetaMean = rep(0, 2)

modelOrderedLogit_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  Y = t(conspiracyItems), 
  nFactors = ncol(Qmatrix),
  Qmatrix = Qmatrix,
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  meanTheta = thetaMean
)


modelOrderedLogit_samples = modelOrderedLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 191120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelOrderedLogit_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelOrderedLogit_samples$summary(variables = c("lambda", "mu", "thetaCorr")) ,n=Inf)

# correlation posterior distribution
mcmc_trace(modelOrderedLogit_samples$draws(variables = "thetaCorr[1,2]"))
mcmc_dens(modelOrderedLogit_samples$draws(variables = "thetaCorr[1,2]"))

# example theta posterior distributions

thetas = modelOrderedLogit_samples$summary(variables = c("theta"))
mcmc_pairs(x = modelOrderedLogit_samples$draws(variables = c("theta[1,1]", "theta[1,2]")))

plot(x = thetas$mean[1:177], y = thetas$mean[178:354], xlab="Theta 1", ylab="Theta 2")

save.image(file = "lecture04e.RData")
