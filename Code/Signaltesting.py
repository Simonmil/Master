import numpy as np
import matplotlib.pyplot as plt
import george
import ROOT as r
from iminuit import Minuit
import time

R = r.TRandom(0)



def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)
    #return pars[2]*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)


class log_like_gp:
    def __init__(self,x,y):
        self.x = x  
        self.y = y

    def __call__(self,Amp,sigma,mass0):
        kernel = Amp * george.kernels.LocalGaussianKernel(location=mass0,log_width=sigma)
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf
    
def fit_minuit_gp(num,lnprob):
    minLLH = np.inf
    best_fit_parameters = (0,0,0)
    for i in range(num):
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=1,sigma=2,mass0=125,
                    error_Amp=0.01,error_sigma=0.01,error_mass0=0.1)#,
                    #fix_sigma=True,fix_mass0=True)

    m.migrad()
    if m.fval < minLLH:
        minLLH = m.fval
        best_fit_parameters = m.args

    print("min LL",minLLH)
    print("Best fit parameters",best_fit_parameters)
    print("sqrt(Amp)",np.sqrt(best_fit_parameters[0]))
    print("Sigma**2",best_fit_parameters[1]*np.sqrt(2))
    
    return minLLH, best_fit_parameters


mean = 125.
sigma = 2.
Amp = 2000.


signal_def = r.TF1("signal_def","gaus",110,160,3)
signal_cust = r.TF1("signal_cust",Gaussian,110,160,3)

signal_def.SetParameters(Amp,mean,sigma)
signal_def.SetParNames("Amplitude","Mean","Sigma")
signal_cust.SetParameters(mean,sigma,Amp)
signal_cust.SetParNames("Mean","Sigma","Amplitude")

Nbins = 100
mass = np.linspace(110,160,Nbins)
toy = np.zeros(Nbins)
h_toy = r.TH1D("h_toy","Signal",Nbins,110,160)

for i in range(100):
    #toy[i] = signal_cust(mass[i])
    toy[i] = signal_def(mass[i])
    h_toy.SetBinContent(i+1,toy[i])
    h_toy.SetBinError(i+1,np.sqrt(toy[i]))


print(h_toy.Integral())
#h_toy.Draw("pe")


lnprob = log_like_gp(mass,toy)
minLL, best_fit_parameters = fit_minuit_gp(100,lnprob)
kernel = best_fit_parameters[0] * george.kernels.LocalGaussianKernel(location=best_fit_parameters[2],log_width=best_fit_parameters[1])
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(mass,yerr=np.sqrt(toy))

y_pred, y_var = gp.predict(toy,mass,return_var=True)

plt.scatter(mass,toy,marker='.',color='r')
plt.plot(mass,y_pred,'b-')
plt.show()

