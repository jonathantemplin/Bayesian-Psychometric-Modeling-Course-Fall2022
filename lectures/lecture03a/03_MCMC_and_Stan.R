rm(list = ls())

# installing rstan (older interface to stan)
if (!require(rstan)) install.packages("rstan")
library(rstan)

# installing cmdstanr
if (!require(cmdstanr)) install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
library(cmdstanr)

# bayesplot: for plotting posterior distributions
if (!require(bayesplot)) install.packages("bayesplot")
library(bayesplot)

# HDInterval: for constructing Highest Density Posterior Intervals
if (!require(HDInterval)) install.packages("HDInterval")
library(HDInterval)

if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)


DietData = read.csv(file = "DietData.csv")

ggplot(data = DietData, aes(x = WeightLB)) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 


ggplot(data = DietData, aes(x = WeightLB, color = factor(DietGroup), fill = factor(DietGroup))) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 


ggplot(data = DietData, aes(x = HeightIN, y = WeightLB, shape = factor(DietGroup), color = factor(DietGroup))) +
  geom_smooth(method = "lm", se = FALSE) + geom_point()

# linear model section

# full analysis model suggested by data:
FullModel = lm(formula = WeightLB ~ 1, data = DietData)

# examining assumptions and leverage of fit
plot(FullModel)

# looking at ANOVA table
anova(FullModel)

# looking at parameter summary
summary(FullModel)

# working with stan:

# show empty model using OLS
emptyModel = lm(formula = WeightLB ~ 1, data = DietData)
summary(emptyModel)
anova(emptyModel)

# Stan syntax (also in model00.stan)
stanModel = "

data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real beta0;
  real<lower=0> sigma;
}

model {
  beta0 ~ normal(0, 1000); // prior for beta0
  sigma ~ uniform(0, 100000); // prior for sigma
  y ~ normal(beta0, sigma);
}

"

# compile model -- this method is for stand-alone stan files
model00.fromFile = cmdstan_model(stan_file = "model00.stan")

# show location of executable:
model00.fromFile$exe_file()

# or this method
model00.fromString = cmdstan_model(stan_file = write_stan_file(stanModel))

# show location of executable:
model00.fromString$exe_file()

# build R list containing data for Stan: Must be named what "data" are listed in analysis
stanData = list(
  N = nrow(DietData),
  y = DietData$WeightLB
)


# run MCMC chain (sample from posterior)
model00.samples = model00.fromFile$sample(
  data = stanData,
  seed = 1,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10000,
  iter_sampling = 10000
)

# example MCMC analysis in rstan
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

model00.rstan = stan(
  model_code = stanModel,
  model_name = "Empty model",
  data = stanData,
  warmup = 10000,
  iter = 20000,
  chains = 4,
  verbose = TRUE
)


# check for convergence

# R-hat
model00.samples

# plots
mcmc_trace(model00.samples$draws(c("beta0", "sigma")))
mcmc_dens(model00.samples$draws(c("beta0", "sigma")))

# examples of non-converged chains

model00.badSamples = model00.fromFile$sample(
  data = stanData,
  seed = 2,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10,
  iter_sampling = 100
)
mcmc_trace(model00.badSamples$draws(c("beta0", "sigma")))
mcmc_dens(model00.badSamples$draws(c("beta0", "sigma")))

# next, summarize parameters
hdi(model00.samples$draws("beta0"))
hdi(model00.samples$draws("sigma"))


save.image("lecture03.RData")
