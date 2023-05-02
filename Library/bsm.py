import numpy as np
import scipy.stats as stats
from math import log, sqrt, exp
import math
import pandas as pd
from scipy.stats import norm
import datetime
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, fsolve
import statsmodels.api as sm

# Black-Scholes formula for European options
def black_scholes(S, K, T, r, q, sigma, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

# Function to find the implied volatility
def implied_volatility(S, K, T, r, q, market_price, option_type):
    def objective_function(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type) - market_price

    try:
        result = brentq(objective_function, 1e-6, 1, full_output=False, disp=False)
    except ValueError:
        result = np.nan
    return result

#greeks
def gbsm_greeks(S, K, t, T, r, q, sigma, option_type='call'):
    tau = T - t
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    if option_type == 'call':
        delta = exp(-q * tau) * norm.cdf(d1)
        gamma = exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt(tau))
        vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        theta = -S * np.exp(q * tau) * norm.pdf(d1) * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * norm.cdf(-d2) + q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = -exp(-q * tau) * (1 - norm.cdf(d1))
        gamma = exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt(tau))
        vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        theta = -S * np.exp(q * tau) * norm.pdf(d1) * sigma / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * norm.cdf(-d2) - q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price, delta, gamma, vega, theta, rho


# Binomial tree
# binomial tree without dividend
def bt_american(underlying,strike,ttm,rf,b,ivol,N,call):
    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    if call:
        z=1
    else:
        z=-1
    # calculate the number of nodes
    def nNode(n):
        return int((n+1)*(n+2)/2)
    # Calculate the index
    def idx(i,j):
        return nNode(j-1)+i
    nNodes = nNode(N)
    optionvalues = np.zeros(nNodes)
    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            index = idx(i,j)
            price = underlying*u**i*d**(j-i)
            optionvalues[index]=max(0,z*(price-strike))
            if j<N:
               optionvalues[index] = max(optionvalues[index],df*(pu*optionvalues[idx(i+1,j+1)]+pd*optionvalues[idx(i,j+1)]))
            # print(i,j,optionvalues[index])
    return optionvalues[0]

# binomial tree with dividend
# divtimes = int((div_date-curr_date).days/(expiration-curr_date).days *N)
def bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call):
    # No dividends
    if len(divamts)==0 or len(divtimes)==0:
        return bt_american(underlying,strike,ttm,rf,rf,ivol,N,call)
    # First div outside grid
    if divtimes[0]>N:
        return bt_american(underlying,strike,ttm,rf,rf,ivol,N,call)
    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(rf*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    if call:
        z=1
    else:
        z=-1
    # calculate the number of nodes
    def nNode(n):
        return int((n+1)*(n+2)/2)
    # Calculate the index
    def idx(i,j):
        return nNode(j-1)+i
    nDiv = len(divtimes)
    nNodes = nNode(divtimes[0])
    optionvalues = np.zeros(nNodes)
    for j in range(divtimes[0],-1,-1):
        for i in range(j,-1,-1):
            index = idx(i,j)
            price = underlying*u**i*d**(j-i)
            if j < divtimes[0]:
                # Times before dividend, backward method
                optionvalues[index]=max(0,z*(price-strike))
                optionvalues[index] = max(optionvalues[index],df*(pu*optionvalues[idx(i+1,j+1)]+pd*optionvalues[idx(i,j+1)]))
            else:
                valnoex = bt_american_div(price-divamts[0],strike,ttm-divtimes[0]*dt,rf,divamts[1:nDiv-1],divtimes[1:nDiv-1]-divtimes[0],ivol,N-divtimes[0],call)
                valex = max(0,z*(price-strike))
                optionvalues[index] = max(valnoex,valex)
                # print("new",i,j,optionvalues[index])
    return optionvalues[0]

def ivol_bt(underlying,strike,ttm,rf,divamts,divtimes,N,call,value,initvol):
    def sol_vol(x,underlying,strike,ttm,rf,divamts,divtimes,N,call,value):
        return bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,x,N,call)-value
    vol = fsolve(sol_vol,initvol,args=(underlying,strike,ttm,rf,divamts,divtimes,N,call,value))
    return vol[0]