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

# Original Graded Response Model=============================================================

modelGRM_standardizedFactor_syntax = "

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

modelGRM_standardizedFactor_stan = cmdstan_model(stan_file = write_stan_file(modelGRM_standardizedFactor_syntax))

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


modelGRM_standardizedFactor_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  Y = t(conspiracyItems), 
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP
)

modelGRM_standardizedFactor_samples = modelGRM_standardizedFactor_stan$sample(
  data = modelGRM_standardizedFactor_data,
  seed = 201120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelGRM_standardizedFactor_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelGRM_standardizedFactor_samples$summary(variables = c("lambda", "mu")) ,n=Inf)

# GRM + Factor Variance: Strong identification (model/data likelihood and posterior) ===========================

modelGRM_markerItem_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory;                     // maximum category across all items
  
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  array[nItems] vector[maxCategory-1] meanThr;                    // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;      // prior covariance matrix for intercept parameters
  
  vector[nItems-1] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems-1, nItems-1] covLambda;  // prior covariance matrix for discrimination parameters
  
  real thetaSDmean;  // prior mean hyperparameter for theta standard deviation (log normal distribution)
  real thetaSDsd; // prior sd hyperparameter for theta standard deviation (log normal distribution)
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  real<lower=0> thetaSD;
  
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nItems-1] initLambda;             // the estimated factor loadings (number of items-1 for one marker item)
  
  
}

transformed parameters{
  vector[nItems] lambda;   // the loadings that go into the model itself
  
  lambda[1] = 1.0;         // first loading on the factor is set to one for identification (marker item)
  lambda[2:(nItems)] = initLambda[1:(nItems-1)]; // rest of loadings are set to estimated values in initLambda
}

model {
  
  initLambda ~ multi_normal(meanLambda, covLambda); // Prior for estimated item discrimination/factor loadings
  thetaSD ~ lognormal(thetaSDmean,thetaSDsd);               // Prior for theta standard deviation
  theta ~ normal(0, thetaSD);                       // Prior for latent variable (with sd specified)
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);             // Prior for item thresholds
    Y[item] ~ ordered_logistic(lambda[item]*theta, thr[item]);         // Item repsonse model (model/data likelihood)
  }
  
  
}

generated quantities{
  array[nItems] vector[maxCategory-1] mu;
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
}

"

modelGRM_markerItem_stan = cmdstan_model(stan_file = write_stan_file(modelGRM_markerItem_syntax))

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
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems-1)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems-1)


modelGRM_markerItem_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  Y = t(conspiracyItems), 
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP, 
  thetaSDmean = 0,
  thetaSDsd = 2
)

modelGRM_markerItem_samples = modelGRM_markerItem_stan$sample(
  data = modelGRM_markerItem_data,
  seed = 201120223,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=10, sd=1))
)

# checking convergence
max(modelGRM_markerItem_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelGRM_markerItem_samples$summary(variables = c("thetaSD", "lambda", "mu")) ,n=Inf)

mcmc_trace(x = modelGRM_markerItem_samples$draws(variables = c("thetaSD")))
mcmc_dens(x = modelGRM_markerItem_samples$draws(variables = c("thetaSD")))

# comparing theta estimates

plot(density(modelGRM_markerItem_samples$summary(variables = "theta")$mean),
     ylim = c(0, max(density(modelGRM_standardizedFactor_samples$summary(variables = "theta")$mean)$y)),
     col = 2, lwd=3, main="Comparing Theta EAP Densities")
lines(density(modelGRM_standardizedFactor_samples$summary(variables = "theta")$mean),
      col=3, lwd =3)
legend(x = 4, y = .4, legend = c("Marker Item", "Standardized Factor"), col=c(2, 3), lwd=c(2,3), lty=c(1,1))

plot(x = modelGRM_markerItem_samples$summary(variables = "theta")$mean,
     y = modelGRM_standardizedFactor_samples$summary(variables = "theta")$mean, 
     ylab = "Standardized Factor", xlab = "Marker Item", main = "Comparing Theta EAP Estimates")

plot(x = modelGRM_markerItem_samples$summary(variables = "theta")$sd,
     y = modelGRM_standardizedFactor_samples$summary(variables = "theta")$sd, 
     ylab = "Standardized Factor", xlab = "Marker Item", main = "Comparing Theta SD Estimates")

# comparing lambda estimates
plot(x = modelGRM_markerItem_samples$summary(variables = "lambda")$mean,
     y = modelGRM_standardizedFactor_samples$summary(variables = "lambda")$mean, 
     ylab = "Standardized Factor", xlab = "Marker Item", main = "Comparing Lambda EAP Estimates")


modelGRM_standardizedFactor_samples$summary(variables = "lambda[1]")
modelGRM_markerItem_samples$summary(variables = "thetaSD")
modelGRM_markerItem_samples$summary(variables = "lambda[1]")



# Multidimesnional GRM + Factor Variance: Strong identification (model/data likelihood and posterior) ===========================


modelMultidimensionalGRM_markerItem_syntax = "
data {
  
  // data specifications  =============================================================
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory;                     // number of categories for each item
  
  // input data  =============================================================
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  // loading specifications  =============================================================
  int<lower=1> nFactors;                                       // number of loadings in the model
  array[nItems, nFactors] int<lower=0, upper=1> Qmatrix;
  
  // prior specifications =============================================================
  array[nItems] vector[maxCategory-1] meanThr;                // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;  // prior covariance matrix for intercept parameters
  
  vector[nItems-nFactors] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems-nFactors, nItems-nFactors] covLambda;  // prior covariance matrix for discrimination parameters
  
  vector[nFactors] meanTheta; 
  vector[nFactors] sdThetaLocation;
  vector[nFactors] sdThetaScale;
}

transformed data{
  int<lower=0> nLoadings = 0;                                      // number of loadings in model
  array[nFactors] int<lower=0> markerItem = rep_array(0, nFactors);
  
  for (factor in 1:nFactors){
    nLoadings = nLoadings + sum(Qmatrix[1:nItems, factor]);
  }

  array[nLoadings, 4] int loadingLocation;                     // the row/column positions of each loading, plus marker switch
  
  int loadingNum=1;
  int lambdaNum=1;
  for (item in 1:nItems){
    for (factor in 1:nFactors){       
      if (Qmatrix[item, factor] == 1){
        loadingLocation[loadingNum, 1] = item;
        loadingLocation[loadingNum, 2] = factor;
        if (markerItem[factor] == 0){
          loadingLocation[loadingNum, 3] = 1;     // ==1 if marker item, ==0 otherwise
          loadingLocation[loadingNum, 4] = 0;     // ==0 if not one of estimated lambdas
          markerItem[factor] = item;
        } else {
          loadingLocation[loadingNum, 3] = 0;
          loadingLocation[loadingNum, 4] = lambdaNum;
          lambdaNum = lambdaNum + 1;
        }
        loadingNum = loadingNum + 1;
      }
    }
  }


}

parameters {
  array[nObs] vector[nFactors] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nLoadings-nFactors] initLambda;             // the factor loadings/item discriminations (one for each item)
  
  cholesky_factor_corr[nFactors] thetaCorrL;
  vector<lower=0>[nFactors] thetaSD;
}

transformed parameters{
  matrix[nItems, nFactors] lambdaMatrix = rep_matrix(0.0, nItems, nFactors);
  matrix[nObs, nFactors] thetaMatrix;
  
  // build matrix for lambdas to multiply theta matrix
  
  for (loading in 1:nLoadings){  
    if (loadingLocation[loading,3] == 1){
      lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = 1.0;
    } else {
      lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = initLambda[loadingLocation[loading,4]];
    }
  }
  
  for (factor in 1:nFactors){
    thetaMatrix[,factor] = to_vector(theta[,factor]);
  }
  
}

model {
  
  matrix[nFactors, nFactors] thetaCovL;
  initLambda ~ multi_normal(meanLambda, covLambda); 
  
  thetaCorrL ~ lkj_corr_cholesky(1.0);
  thetaSD ~ lognormal(sdThetaLocation,sdThetaScale);
  
  thetaCovL = diag_pre_multiply(thetaSD, thetaCorrL);
  theta ~ multi_normal_cholesky(meanTheta, thetaCovL);    
  
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);            
    Y[item] ~ ordered_logistic(thetaMatrix*lambdaMatrix[item,1:nFactors]', thr[item]);
  }
  
  
}

generated quantities{ 
  array[nItems] vector[maxCategory-1] mu;
  corr_matrix[nFactors] thetaCorr;
  cholesky_factor_cov[nFactors] thetaCov_pre;
  cov_matrix[nFactors] thetaCov; 
  
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
  
  thetaCorr = multiply_lower_tri_self_transpose(thetaCorrL);
  thetaCov_pre = diag_pre_multiply(thetaSD, thetaCorrL);
  thetaCov = multiply_lower_tri_self_transpose(thetaCov_pre);
}


"

modelMultidimensionalGRM_markerItem_stan = cmdstan_model(stan_file = write_stan_file(modelMultidimensionalGRM_markerItem_syntax))


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


# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)
nFactors = ncol(Qmatrix)

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
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems-nFactors)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems-nFactors)

# theta hyperparameters
thetaMean = rep(0, nFactors)
sdThetaLocation = rep(0, nFactors)
sdThetaScale = rep(.5, nFactors)

modelMultidimensionalGRM_markerItem_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  Y = t(conspiracyItems), 
  nFactors = nFactors,
  Qmatrix = Qmatrix,
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  meanTheta = thetaMean,
  sdThetaLocation = sdThetaLocation,
  sdThetaScale = sdThetaScale
)

modelMultidimensionalGRM_markerItem_samples = modelMultidimensionalGRM_markerItem_stan$sample(
  data = modelMultidimensionalGRM_markerItem_data,
  seed = 201120224,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  thin = 2,
  init = function() list(lambda=rnorm(nItems, mean=10, sd=1))
)

 # checking convergence
max(modelMultidimensionalGRM_markerItem_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelMultidimensionalGRM_markerItem_samples$summary(variables = c("thetaSD", "thetaCov", "thetaCorr", "lambdaMatrix", "mu")) ,n=Inf)

save.image(file = "lecture04h.RData")
