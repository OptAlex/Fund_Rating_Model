# Portfolio Credit Risk Simulation

This project is a sophisticated exploration into the estimation of default probabilities and Credit Value-at-Risk (CVaR) for a portfolio of funds. We leverage a high-level modelling approach that integrates ARMA-GARCH and Copula methods to simulate daily returns for each fund. The simulation process, executed in R, involves generating 252 daily returns for each fund over 1000 iterations. Subsequent calculations of default probabilities and CVaR are performed in Python, with the 'help_functions' folder housing all the essential helper functions and 'main.py' presenting the central sequence of operations.

Our simulation methodology capitalizes on parallel processing, optimizing computational efficiency in accordance with the number of available CPU cores. Acknowledging the potential for numerical instability, we implemented a robust strategy for bootstrapping and simulation. Our chosen configuration involves conducting 10 iterations of bootstrapping with 1000 simulations for each of the 26 funds across 252 days. Throughout this project, we accomplished 580 unique bootstrapping procedures, executing a complete simulation run on each. Post-simulation plotting and analysis are manually executed, and do not form part of the automated process.

# Project Design

Construct a portfolio of funds, comprising more than 20 different funds.
Source daily market data for these funds spanning a minimum of 10 years.
Determine an appropriate ARMA-GARCH + Copula model for these funds.
Simulate the Probability of Default (PD) and 1-Year Credit VaR of this portfolio based on the identified models.
Essential Elements of Implementation

## ARMA-GARCH + Copula Model
The modelling process utilizes the ARMA-GARCH and Copula methods for time-series modelling of each fund's data. The bootstrap_returns function in the 'bootstrap_returns' helper file embodies these methods.

## PD and 1-Year Credit VaR Simulation
The simulation of PD and 1-Year Credit VaR are conducted based on the returns derived from the model.

PD is simulated using the calc_threshold_violation function in the 'calculate_default_probs' helper file. This function estimates the cumulative returns from the bootstrapped simulations and assesses whether they breach a predetermined threshold, which, if violated, signifies a default.

1-Year Credit VaR is evaluated via the get_CVaR function in the 'get_CVaR' helper file. CVaR, also referred to as Expected Shortfall (ES), quantifies the expected loss of an investment under worst-case conditions and is employed to estimate the 1-Year Credit VaR.

# Data Transformation Helpers

## Several helper functions facilitate data transformations and calculations:

input_data formats the data for subsequent operations.
create_log_returns computes the log returns from historical price time series.
get_fund_dict creates a dictionary of all funds with their corresponding returns.
convert_returns transitions between standard returns and log returns.
calculate_portfolio_return calculates the portfolio returns via a weighted average.
Default Probability Helpers

## Additional helper functions for calculating default probabilities include:

calc_threshold_violation assesses whether a fund breaches a threshold.
calc_default_probs calculates default probabilities based on threshold breaches.
Advanced Bootstrapping

## Multivariat Time Series Bootstrapping
An essential component of this project is the application of advanced bootstrapping techniques. Although the technique of univariate bootstrapping for multivariate time series data has been previously discussed in literature (Dimitris N. Politis & Halbert White, 2004), our  contribution is in combining these methods for an average and applying it to multivariate time series data. Empirical results demonstrate the effectiveness of our bootstrapping method, with the unbootstrapped data centrally located within our intervals.

# References

Martin Grziska. (2013). Multivariate GARCH and Dynamic Copula Models for Financial Time Series With an Application to Emerging Markets.

Hofert, Marius. (2023). The Copula GARCH Model.

“Market Risk Analysis, Value at Risk Models” by Carol Alexander · 2009, page 86, modelling equity portfolio VaR.

"Measuring Performance of Exchange Traded Funds" by Marlène Hassine & Thierry Roncalli.

Dimitris N. Politis & Halbert White (2004) Automatic Block-Length Selection for the Dependent Bootstrap, Econometric Reviews, 23:1, 53-70, DOI: 10.1081/ETC-120028836.

# Final Note

While models provide useful approximations of reality, their outputs must be interpreted cautiously, considering their assumptions and potential model risks. Integrating robust model validation, backtesting, and sensitivity analysis is a critical aspect of any modelling process.
