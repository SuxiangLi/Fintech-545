Value at Risk (VaR) and Expected Shortfall (ES) Calculation

This code calculates the VaR and ES for a given portfolio of stocks using historical data and Monte Carlo simulation. The code is divided into three parts:

Problem 1
The first part of the code fits a normal distribution and a Generalized T distribution to the historical returns of a single stock, and then calculates the VaR and ES for each distribution. The VaR and ES are plotted on a histogram of the returns, and the VaR and ES for both distributions are compared.

Problem 2
The second part of the code calculates the covariance matrix of the daily returns for a given set of stocks using an exponentially weighted covariance estimator. The code then applies a non-positive definite (non-PSD) fix to the covariance matrix to ensure that it is positive definite. The Cholesky decomposition is then used to obtain the square root of the covariance matrix, and direct simulation is used to generate 1000 random draws from the multivariate normal distribution with the given mean and covariance.

Problem 3
The final part of the code uses Monte Carlo simulation to calculate the VaR and ES for a given portfolio of stocks. The daily returns of each stock are first standardized, and then a t-distribution is fit to each standardized return series. The code then generates 10,000 simulations of the returns for each stock using the fitted t-distributions. For each simulation, the code calculates the change in the price of each stock in the portfolio, and then calculates the change in the value of the portfolio. The VaR and ES are then calculated using the simulated changes in portfolio value.

Overall, this code provides a comprehensive solution for calculating VaR and ES for a given portfolio of stocks using both historical data and Monte Carlo simulation.
