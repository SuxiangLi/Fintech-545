This Python code covers three main problems related to stock options and portfolio optimization.

Problem 1: Greeks and Valuation of Options
The code calculates the following:

Greeks for European call and put options using the closed-form GBSM (Black-Scholes-Merton) model.
Greeks for European call and put options using finite difference derivatives.
Valuation of American call and put options with and without discrete dividends using the binomial tree model.
The Greeks are calculated for a given stock price, strike price, current date, options expiration date, risk-free rate, and continuously compounding coupon. Additionally, the sensitivity of the put and call options to a change in the dividend amount is determined.

Problem 2: Value at Risk (VaR) and Expected Shortfall (ES) Calculations
The code calculates the VaR and ES for an options portfolio with the following assumptions:

American options
Current AAPL price
Risk-free rate
Dividend payment on a specified date
The calculations are based on fitting a Normal distribution to the daily returns of AAPL stock, simulating 10 days ahead, and applying the simulated returns to the current AAPL price. VaR and ES are also calculated using the Delta-Normal method. All results are presented in terms of dollar loss instead of percentages. The calculated values are then compared to the results from the previous week's analysis.

Problem 3: Portfolio Optimization using Fama-French 4-Factor Model
The code performs the following tasks:

Fits a 4-factor model (Fama-French 3 factors and Carhart's momentum factor) to the returns of a set of 20 stocks.
Calculates the expected annual return for each stock based on the past 10 years of factor returns.
Constructs an annual covariance matrix for the 20 stocks.
Finds the super-efficient portfolio, assuming a given risk-free rate.
The 4-factor model is fitted using the Fama-French 3-factor return time series and Carhart's momentum time series. The expected returns are then used to optimize the portfolio, finding the optimal weights for each stock.
