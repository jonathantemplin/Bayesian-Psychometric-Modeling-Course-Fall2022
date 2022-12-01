# Package installation ======================================================================
needed_packages = c("ggplot2", "cmdstanr", "HDInterval", "bayesplot", "loo", "networkD3")
for(i in 1:length(needed_packages)){
  haspackage = require(needed_packages[i], character.only = TRUE)
  if(haspackage == FALSE){
    install.packages(needed_packages[i])
  }
  library(needed_packages[i], character.only = TRUE)
}
# set number of cores to 4 for this analysis
options(mc.cores = 4)
set.seed(seed = 1)
# Import data ===============================================================================

conspiracyData = read.csv("conspiracies.csv")
conspiracyItems = conspiracyData[,1:10]

# Binomial Model Syntax (slope/intercept form ) ==================================================



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
  
  // for PPMC:
  array[nItems, nObs] int<lower=0> simY;
  
  // for LOO/WAIC:
  vector[nObs] personLike = rep_vector(0.0, nObs);
  
  for (item in 1:nItems){
    for (obs in 1:nObs){
      // generate data based on distribution and model
      simY[item, obs] = ordered_logistic_rng(lambda[item]*theta[obs], thr[item]);
      
      // calculate conditional data likelihood for LOO/WAIC
      personLike[obs] = personLike[obs] + ordered_logistic_lpmf(Y[item, obs] | lambda[item]*theta[obs], thr[item]);
    }
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

# setting up PPMC
simData = modelOrderedLogit_samples$draws(variables = "simY", format = "draws_matrix")
colnames(simData)
dim(simData)


# set up object for storing each iteration's PPMC data
nPairs = choose(10, 2)
pairNames = NULL
for (col in 1:(nItems-1)){
  for (row in (col+1):nItems){
    pairNames = c(pairNames, paste0("item", row, "_item", col))
  }
}

PPMCsamples = list()

PPMCsamples$correlation = NULL
PPMCsamples$mean = NULL

# loop over each posterior sample's simulated data

for (sample in 1:nrow(simData)){
  
  # create data frame that has observations (rows) by items (columns)
  sampleData = data.frame(matrix(data = NA, nrow = nObs, ncol = nItems))
  
  for (item in 1:nItems){
    itemColumns = colnames(simData)[grep(pattern = paste0("simY\\[", item, "\\,"), x = colnames(simData))] 
    sampleData[,item] = t(simData[sample, itemColumns])
  }
  # with data frame created, apply functions of the data:
  
  # calculate item means
  PPMCsamples$mean = rbind(PPMCsamples$mean, apply(X = sampleData, MARGIN = 2, FUN = mean))
  
  # calculate pearson correlations  
  temp=cor(sampleData)
  PPMCsamples$correlation = rbind(PPMCsamples$correlation, temp[lower.tri(temp)])
  
  
}

colnames(PPMCsamples$correlation) = pairNames
colnames(PPMCsamples$mean) = paste0("item", 1:nItems)

# example densities of some statistics
plot(density(PPMCsamples$mean[,1]), main = "Posterior Predictive Distribution: Item 1 Mean")
lines(x = c(mean(conspiracyItems$PolConsp1),mean(conspiracyItems$PolConsp1)), y = c(0,10),
      lty = 2, col=2, lwd=3)

plot(density(PPMCsamples$correlation[,1]), main = "Item 1 Item 2 Pearson Correlation")
lines(x = c(cor(conspiracyItems[,1:2])[1,2],
            cor(conspiracyItems[,1:2])[1,2]),
      y = c(0,10),
      lty = 2, col=2, lwd=3)


# next, build distributions for each type of statistic
meanSummary = NULL

# for means
for (item in 1:nItems){
  
  tempDist = ecdf(PPMCsamples$mean[,item])
  ppmcMean = mean(PPMCsamples$mean[,item])
  observedMean = mean(conspiracyItems[,item])
  meanSummary = rbind(
    meanSummary,
    data.frame(
      item = paste0("Item", item),
      ppmcMean = ppmcMean,
      observedMean = observedMean,
      residual = observedMean - ppmcMean,
      observedMeanPCT = tempDist(observedMean)
    )
  )
  
}
View(meanSummary)

# for pearson correlations
corrSummary = NULL

# for means
for (column in 1:ncol(PPMCsamples$correlation)){
  
  # get item numbers from items
  items = unlist(strsplit(x = colnames(PPMCsamples$correlation)[column], split = "_"))
  item1num = as.numeric(substr(x = items[1], start = 5, stop = nchar(items[1])))
  item2num = as.numeric(substr(x = items[2], start = 5, stop = nchar(items[2])))
  
  tempDist = ecdf(PPMCsamples$correlation[,column])
  ppmcCorr = mean(PPMCsamples$correlation[,column])
  observedCorr = cor(conspiracyItems[,c(item1num, item2num)])[1,2]
  pct = tempDist(observedCorr)
  if (pct > .975 | pct < .025){
    inTail = TRUE
  } else {
    inTail = FALSE
  }
  corrSummary = rbind(
    corrSummary,
    data.frame(
      item1 = paste0("Item", item1num),
      item2 = paste0("Item", item2num),
      ppmcCorr = ppmcCorr,
      observedCorr = observedCorr,
      residual = observedCorr - ppmcCorr,
      observedCorrPCT = pct, 
      inTail = inTail
    )
  )
  
}

View(corrSummary)

# gather count of problematic items
badItems = c(
 corrSummary$item1[corrSummary$inTail],
 corrSummary$item2[corrSummary$inTail]
)
table(badItems)[order(table(badItems), decreasing = TRUE)]

simpleNetwork(corrSummary[ corrSummary$inTail, c(1,2)])

# Multidimensional Model ================================================================


modelOrderedLogit2D_syntax = "
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

  // for PPMC:
  array[nItems, nObs] int<lower=0> simY;
  
  // for LOO/WAIC:
  vector[nObs] personLike = rep_vector(0.0, nObs);
  
  for (item in 1:nItems){
    for (obs in 1:nObs){
      // generate data based on distribution and model
      simY[item, obs] = ordered_logistic_rng(thetaMatrix[obs,]*lambdaMatrix[item,1:nFactors]', thr[item]);
      
      // calculate conditional data likelihood for LOO/WAIC
      personLike[obs] = personLike[obs] + ordered_logistic_lpmf(Y[item, obs] | thetaMatrix[obs,]*lambdaMatrix[item,1:nFactors]', thr[item]);
    }
  }  
}


"


modelOrderedLogit2D_stan = cmdstan_model(stan_file = write_stan_file(modelOrderedLogit2D_syntax))

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

modelOrderedLogit2D_data = list(
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


modelOrderedLogit2D_samples = modelOrderedLogit2D_stan$sample(
  data = modelOrderedLogit2D_data,
  seed = 191120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelOrderedLogit2D_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelOrderedLogit2D_samples$summary(variables = c("lambda", "mu", "thetaCorr")) ,n=Inf)


# setting up PPMC

simData = modelOrderedLogit2D_samples$draws(variables = "simY", format = "draws_matrix")
colnames(simData)
dim(simData)


# set up object for storing each iteration's PPMC data
nPairs = choose(10, 2)
pairNames = NULL
for (col in 1:(nItems-1)){
  for (row in (col+1):nItems){
    pairNames = c(pairNames, paste0("item", row, "_item", col))
  }
}

PPMCsamples2 = list()

PPMCsamples2$correlation = NULL
PPMCsamples2$mean = NULL

# loop over each posterior sample's simulated data

for (sample in 1:nrow(simData)){
  
  # create data frame that has observations (rows) by items (columns)
  sampleData = data.frame(matrix(data = NA, nrow = nObs, ncol = nItems))
  
  for (item in 1:nItems){
    itemColumns = colnames(simData)[grep(pattern = paste0("simY\\[", item, "\\,"), x = colnames(simData))] 
    sampleData[,item] = t(simData[sample, itemColumns])
  }
  # with data frame created, apply functions of the data:
  
  # calculate item means
  PPMCsamples2$mean = rbind(PPMCsamples2$mean, apply(X = sampleData, MARGIN = 2, FUN = mean))
  
  # calculate pearson correlations  
  temp=cor(sampleData)
  PPMCsamples2$correlation = rbind(PPMCsamples2$correlation, temp[lower.tri(temp)])
  
  
}

colnames(PPMCsamples2$correlation) = pairNames
colnames(PPMCsamples2$mean) = paste0("item", 1:nItems)

# example densities of some statistics
plot(density(PPMCsamples2$mean[,1]), main = "Posterior Predictive Distribution: Item 1 Mean")
lines(x = c(mean(conspiracyItems$PolConsp1),mean(conspiracyItems$PolConsp1)), y = c(0,10),
      lty = 2, col=2, lwd=3)

plot(density(PPMCsamples2$correlation[,1]), main = "Item 1 Item 2 Pearson Correlation")
lines(x = c(cor(conspiracyItems[,1:2])[1,2],
            cor(conspiracyItems[,1:2])[1,2]),
      y = c(0,10),
      lty = 2, col=2, lwd=3)


# next, build distributions for each type of statistic
meanSummary2 = NULL

# for means
for (item in 1:nItems){
  
  tempDist = ecdf(PPMCsamples2$mean[,item])
  ppmcMean = mean(PPMCsamples2$mean[,item])
  observedMean = mean(conspiracyItems[,item])
  meanSummary2 = rbind(
    meanSummary2,
    data.frame(
      item = paste0("Item", item),
      ppmcMean = ppmcMean,
      observedMean = observedMean,
      residual = observedMean - ppmcMean,
      observedMeanPCT = tempDist(observedMean)
    )
  )
  
}
View(meanSummary2)

# for  correlations
corrSummary2 = NULL

# for means
for (column in 1:ncol(PPMCsamples2$correlation)){
  
  # get item numbers from items
  items = unlist(strsplit(x = colnames(PPMCsamples2$correlation)[column], split = "_"))
  item1num = as.numeric(substr(x = items[1], start = 5, stop = nchar(items[1])))
  item2num = as.numeric(substr(x = items[2], start = 5, stop = nchar(items[2])))
  
  tempDist = ecdf(PPMCsamples2$correlation[,column])
  ppmcCorr = mean(PPMCsamples2$correlation[,column])
  observedCorr = cor(conspiracyItems[,c(item1num, item2num)])[1,2]
  pct = tempDist(observedCorr)
  if (pct > .975 | pct < .025){
    inTail = TRUE
  } else {
    inTail = FALSE
  }
  corrSummary2 = rbind(
    corrSummary2,
    data.frame(
      item1 = paste0("Item", item1num),
      item2 = paste0("Item", item2num),
      ppmcCorr = ppmcCorr,
      observedCorr = observedCorr,
      residual = observedCorr - ppmcCorr,
      observedCorrPCT = pct, 
      inTail = inTail
    )
  )
  
}

View(corrSummary2)

corrSummaryBoth = merge(corrSummary, corrSummary2, by=c("item1", "item2", "observedCorr"))

# gather count of problematic items
badItems2= c(
  corrSummary2$item1[corrSummary2$inTail],
  corrSummary2$item2[corrSummary2$inTail]
)
table(badItems2)[order(table(badItems2), decreasing = TRUE)]

simpleNetwork(corrSummary2[ corrSummary2$inTail, c(1,2)])

# model comparisons
waic(x = modelOrderedLogit_samples$draws("personLike"))
waic(x = modelOrderedLogit2D_samples$draws("personLike"))

modelOrderedLogit_samples$loo(variables = "personLike")
modelOrderedLogit2D_samples$loo(variables = "personLike")

loo_compare(list(unidimensional = modelOrderedLogit_samples$loo(variables = "personLike"), 
                 twodimensional = modelOrderedLogit2D_samples$loo(variables = "personLike")))


save.image(file = "lecture05.RData")
