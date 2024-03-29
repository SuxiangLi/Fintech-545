import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd, eigh
from numpy.linalg import inv, LinAlgError

# Cholesky Factorization
def chol_psd(A):
  n = A.shape[1]
  root = np.zeros((n,n))
  
  # loop over columns
  for i in range(n):
    s = 0.0
    if i > 0:
      s = root[i][:i].T @ root[i][:i]
    
    # Diagonal Element
    temp = A[i][i] - s
    if temp <= 0 and temp >= -1e-8:
      temp = 0.0
    root[i][i] = np.sqrt(temp)

    # check for the 0 eign value. set the column to 0 if we have one
    if root[i][i] == 0.0:
      root[i][(i+1):n] = 0.0
    else:
      # update off diagonal rows of the column
      ir = 1.0/root[i][i]
      for j in np.arange(i+1,n):
        s = root[j][:i].T @ root[i][:i]
        root[j][i] = (A[j][i] -s) * ir
  return root

# direct simulation
def direct_simulation(cov, num):
  result = chol_psd(cov) @ np.random.standard_normal(size=(len(cov), num))
  return result

num = 25000

# PCA simulation
def simulate_pca(a, nsim, perc):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eig(a)

    # flip the eigenvalues and the vectors
    flip = np.argsort(vals)[::-1]
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = np.sum(vals)
    start = 0
    while (np.abs(np.sum(vals[:start])/tv) <perc):
      start+=1
    vals = vals[:start]
    vecs = vecs[:, :start]
    print("Simulating with", start, "PC Factors: {:.2f}".format(np.abs(sum(vals)/tv*100)), "% total variance explained")
    B = np.matmul(vecs, np.diag(np.sqrt(vals)))
    m = B.shape[1]
    r = np.random.randn(m,nsim)
    return np.matmul(B, r)

