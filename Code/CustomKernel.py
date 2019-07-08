import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Kernel, _check_length_scale, Hyperparameter, NormalizedKernelMixin, StationaryKernelMixin



class ExponentialKernel(StationaryKernelMixin,NormalizedKernelMixin,Kernel):
    """Exponential kernel.

    A simple exponential function. 

    Parameters
    -------------
    a and d.
    """

    def __init__(self,a = 1.0, a_bounds=(1e-5,1e5), d = 1.0, d_bounds=(1e-5,1e5)):
        self.a = a
        self.a_bounds = a_bounds
        self.d = d
        self.d_bounds = d_bounds
    

    @property
    def anisotropic(self):
        return np.iterable(self.a) and len(self.a) > 1
    
    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("a","numeric",self.a_bounds,len(self.a))

        return Hyperparameter("a","numeric",self.a_bounds)

    def __call__(self,X,Y=None,eval_gradient=False):
        """ Return the kernel k(X,Y) and optionally its gradient.

        """

        X = np.atleast_2d(X)
        a = _check_length_scale(X,self.a)
        if Y is None:
            dists = pdist(X/a,metric = 'sqeuclidean')
            K = np.exp(-0.5*dists)
            K = squareform(K)
            np.fill_diagonal(K,1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None")
            dists = cdist(X/a,Y/a,metric='sqeuclidean')
            K = np.exp(-0.5*dists)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K,np.empty((X.shape[0],X.shape[0],0))
            elif not self.anisotropic or a.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:,:,np.newaxis]
                return K,K_gradient
            elif self.anisotropic:
                K_gradient = (X[:,np.newaxis,:] - X[np.newaxis,:,:])**2/(a**2)
                K-gradient *= K[...,np.newaxis]
                return K,K_gradient
        
        else:
            return K
    
    def __repr__(self):
        if self.anisotropic:
            return "{0}(a=[{1}]".format(self.__class__.__name__,",".join(map("{0:.3g}".format,self.a)))
        else:
            return "{0}(a={1:.3g})".format(self.__class__.__name__,np.ravel(self.a)[0])



