import numpy as np
import matplotlib.pyplot as plt
import george
import ROOT
#from iminuit import minimize
from scipy.optimize import minimize,fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def underlying_function(x,p0,p1,p2):
    return np.exp(p0 + p1 * (x - 100) + p2 * (x - 100) * (x - 100))

def neg_log_like(p):
    print(p)
    gp.set_parameter_vector(p)
    print(-gp.log_likelihood(toy))
    return -gp.log_likelihood(toy)

def grad_neg_log_like(p):
    gp.set_parameter_vector(p)
    #print(-gp.grad_log_likelihood(toy))
    return -gp.grad_log_likelihood(toy)

p0 = 5
p1 = -0.03
p2 = 1e-6

mass = np.linspace(100,199,100)
truth = underlying_function(mass,p0,p1,p2)

kernel = 1.0*george.kernels.ExpSquaredKernel(metric=1.0)
kernel_sklearn = 1.0*RBF(length_scale=1.0)#, length_scale_bounds=(0.5,1.5))
Ntoys = 1

toy = np.random.poisson(truth)

chi2_ndf = np.zeros(Ntoys)


"""
def get_log(X,Y):
    Z = np.zeros((len(X),len(Y)))
    for i in range(0,len(X)):
        for j in range(0,len(Y)):
            k = np.exp(X[j][i])*george.kernels.ExpSquaredKernel(metric=np.exp(Y[j][i]))#, block=(-0.05,1.5))
            k = X[j][i]*RBF(length_scale=Y[j][i],length_scale_bounds=(0.5,5))
            #gp = george.GP(k, solver=george.HODLRSolver)
            #gp.compute(mass,yerr=1e-2)
            gp = GaussianProcessRegressor(kernel=k,alpha=np.sqrt(toy))
            gp.fit(mass.reshape(-1,1),toy)
            
            Z[j][i] = gp.log_marginal_likelihood()
            print(Z[j][i])
    
    return Z

amp = np.linspace(1,20,20)
l = np.linspace(0.5,5,20)

X,Y = np.meshgrid(amp,l)
Z = get_log(X,Y)

im = plt.contour(X,Y,Z,20)
plt.colorbar(im,orientation='horizontal',shrink=0.8)
row,col = np.where(Z==np.max(Z))
plt.plot(X[0][col[0]],Y[row[0]][0],'b+')
plt.show()

foobar

"""

toy = np.random.poisson(truth)

#plt.plot(mass,toy)
#plt.show()

mass_pred = np.linspace(100,199,1000)
#foobar
#print("Done")
gp = george.GP(kernel)
#print("George params:", gp.get_parameter_vector())
gp.compute(mass,yerr=np.sqrt(toy))

#m = fmin_l_bfgs_b(neg_log_like,gp.get_parameter_vector(),fprime=grad_neg_log_like)
#gp.set_parameter_vector(gp_sklearn.kernel_.theta)
m = minimize(neg_log_like,gp.get_parameter_vector(),jac=grad_neg_log_like)
gp.set_parameter_vector(m.x)
print(m)
#kernel_sklearn = gp.get_parameter_vector()[0]*RBF(length_scale=gp.get_parameter_vector()[1])
gp_sklearn = GaussianProcessRegressor(kernel=kernel_sklearn, alpha=np.sqrt(toy),n_restarts_optimizer=20)

gp_sklearn.fit(mass.reshape(-1,1),toy)
#print("Sklearn params:", gp_sklearn.kernel_.theta)
y_mean = gp_sklearn.predict(mass_pred.reshape(-1,1))
y_pred, y_var = gp.predict(toy,mass_pred,return_cov=False,return_var=True)
plt.plot(mass,toy,color='k',ls='--')
plt.plot(mass_pred,y_pred,color='b')
plt.plot(mass_pred,y_mean,color='r')
plt.fill_between(mass_pred,y_pred-np.sqrt(y_var),y_pred+np.sqrt(y_var),alpha=0.4)
print("Sklearn params:", gp_sklearn.kernel_.theta)
print("Sklearn log-like:", gp_sklearn.log_marginal_likelihood())
print("George params:", gp.get_parameter_vector())
print("George log-like:", gp.log_likelihood(toy))
plt.show()


#chi2 = np.sum((toy - y_pred)**2/toy)
#chi2_ndf[t] = chi2/(len(toy) - 1 - len(gp.get_parameter_vector()))

#plt.hist(chi2_ndf)
#plt.show()

