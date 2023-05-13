# Loading all the necessary libraries
# install.packages("copula")
# install.packages("rugarch")
# install.packages("nortest")
# install.packages("MASS")
# install.packages("stats")
# install.packages("forecast")

require(copula)
require(rugarch)
library(nortest)
library(MASS)
library(stats)
library(forecast)


# Data loading, cleaning, and defininf dimensions and no. of simulations
setwd("/Users/alexander/PycharmProjects/marketRisk/data")

d <- ncol(data)
n <- 252

# Store column names of data
column_names <- colnames(data)

# Fit ARMA + GARCH model
uspec <- ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
                    mean.model = list(armaOrder = c(1, 1)), distribution.model = "std") # Specifying the model parameters

# Creating a fit object that fits the ARMA + GARCH to all our columns
fit <- apply(data, 2, function(x) ugarchfit(uspec, data = x, solver = 'hybrid'))

# Collecting and standardizing the residuals
hist_resid_stand <- sapply(fit, residuals, standardize = TRUE) # Collecting and standardizing the residuals
hist_resid_empir <- pobs(hist_resid_stand) # pobs() is in this case used to calculate the empirical probabilities of the standardized residuals
                             # to fit to the copula

# Creating the fit copula parameter to fit to the copula
fitcop <- fitCopula(ellipCopula("t", dim = d), data = hist_resid_empir, method = "mpl", optim.method = "SANN")
df <- fitcop@estimate[2]

# Creating the object of fitted copula residuals 
sim_resid <- rCopula(n, fitcop@copula)

# Plotting the historical vs simulated residuals from the copula
# plot(hist_resid_empir[,1:2], xlab = expression(hat(U)[1]), ylab = expression(hat(U)[2]), col = 'blue')
# points(sim_resid[,1:2], xlab = expression(hat(U)[sim]), ylab = expression(hat(U)[sim]), col = 'red')

# Defining the df based on the fitcop params and standardizing the simulated residuals
nu. = rep(df, d)
sim_resid_stand <- sapply(1:d, function(j) sqrt((nu.[j]-2)/nu.[j]) * qt(sim_resid[,j], df = nu.[j]))

# Standardizing the simulated residuals from the gaussian copula
# sim_resid_stand <- qnorm(sim_resid)

# Feeding the simulated residuals back to the ARMA + GARCH model
sim <- lapply(1:d, function(j)
  ugarchsim(fit[[j]], n.sim = n, m.sim = 1,
            custom.dist = list(name = "sample",
                               distfit = sim_resid_stand[,j, drop = FALSE])))

# Obtaining the simulated returns
simulated_returns <- sapply(sim, function(x) fitted(x)) # simulated series X_t (= x@simulation$seriesSim)
# matplot(simulated_returns, type = "l", xlab = "t", ylab = expression(X[t]))

# Cumulating the simulated returns
#simulated_returns_cumsum <- apply(simulated_returns, 2, function(x) cumsum(x))
# matplot(simulated_returns_cumsum, type = "l", xlab = "t", ylab = expression(X[t_cumsum]))

# Plotting the standardized historical residuals vs the standardized simulated residuals
# plot(hist_resid_stand[,1:2], xlab = expression(hat(U)[1]), ylab = expression(hat(U)[2]), col = 'blue')
# points(sim_resid_stand[,1:2], xlab = expression(hat(U)[sim]), ylab = expression(hat(U)[sim]), col = 'red')

# Transforming the column names to TICKERS of each fund respectively
colnames(simulated_returns) = column_names

# plot(data[,1:2], xlab = expression(hat(U)[1]), ylab = expression(hat(U)[2]), col = 'blue')
# points(simulated_returns[,1:2], xlab = expression(hat(U)[sim]), ylab = expression(hat(U)[sim]), col = 'red')

# Writing the output data to a csv format for further computation
simulated_returns <- as.data.frame(simulated_returns)  # Convert to data frame if necessary

# Writing the output data to a csv format for further computation
# write.csv(simulated_returns, file = "sim_returns_df.csv", row.names = FALSE)