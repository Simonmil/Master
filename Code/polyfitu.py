"""
Copied from Alex Read
"""

import numpy as np


def polyfitu(x,y,dy,N):
    
    
    #n = len(y) # number of data points
    dysq = dy**2 # Speed up computation
    wtsq = 1./dysq # Speed up computation

    # Form the vector of the Sigma

    kk = np.linspace(1,N+1,N+1).reshape(-1,1)
    Sigma_y = sum(y*x**(kk-1)*wtsq,1)
    

    Sigma_xx = np.zeros((N+1,N+1))
    for k in range(1,N+1):
        for j in range(k,N+1):
            Sigma_xx[j,k] = np.sum(x**(k-1)*x**(j-1)*wtsq)
            if k==j:
                Sigma_xx[k][j] = Sigma_xx[j][k]



    Sigma_xx_inv = np.inv(Sigma_xx)

    p = Sigma_y*Sigma_xx_inv


    xvector = p.zeros((N+1,n))
    xvector[kk,:] = x**(kk-1)*wtsq

    dpdy = Sigma_xx_inv*xvector



    CM = npzeros((N+1,N+1))
    for k in range(1,N+1):
        for j in range(k,N+1):
            CM[j,k] = np.sum(dpdy[k,:]*dpdy[j,:]*dysq)
            if k==j:
                CM[k,j] = CM[j,k]
    

    dp = np.sqrt(np.diag(CM))

