# -*- coding: utf-8 -*-
"""VaR

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_8ci_L4w5pe359i0i4hHHGsA8uH5zTqB
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg

# Calculate Var
def calculate_var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)

# Calculate ES
def calculate_es(data, var, alpha=0.05):
  return data.expect(lambda x: x, lb=-np.inf, ub=-var)/alpha

# Define portfolio size and confidence level
#portfolio_size = 1_000_000
confidence_level = 0.95

# 1. VaR using normal distribution
def var_normal(returns):
  std_dev = returns.std()
  z_score = stats.norm.ppf(confidence_level)
  var_normal = -(std_dev * z_score)
  return var_normal

# 2. VaR using normal distribution with EWM variance (lambda = 0.94)
def var_normal_ewm(returns):
  variance = returns.ewm(alpha=0.06).var().iloc[-1]
  std_dev_ewm = np.sqrt(variance)
  z_score_ewm = stats.norm.ppf(confidence_level)
  var_ewm = -(std_dev_ewm * z_score_ewm)
  return var_ewm

# 3. VaR using MLE fitted T distribution
def var_t(returns):
  params = stats.t.fit(returns)
  t_dist = stats.t(*params)
  var_tdist = -t_dist.ppf(confidence_level)
  return var_tdist

# 4. VaR using fitted AR(1) model
def var_AR1(returns):
  model = sm.tsa.AR(returns)
  results = model.fit(maxlag=1)
  rho = results.params[1]
  var_ar1 = -(rho * returns.mean() + np.sqrt(1 - rho ** 2) * returns.std() * stats.norm.ppf(confidence_level))
  return var_ar1

# 5. VaR using Historic Simulation
def var_historic(returns):
  data_sorted = returns.sort_values()
  var_hist = -data_sorted.quantile(1 - confidence_level)
  return var_hist

