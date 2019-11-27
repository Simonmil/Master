import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
import george
from iminuit import Minuit
from scipy.interpolate import BPoly


R = r.TRandom(0)


def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])

def Gaussian(vars,pars):
    return pars[0]/(np.sqrt(2*np.pi)*pars[2])*np.exp(-0.5*((vars[0]-pars[1])/pars[2])**2)

class log_like_gp:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __call__(self,Amp,length):
        kernel_bkg = Amp * george.kernels.ExpSquaredKernel(metric=length)
        kernel = kernel_bkg
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf

def fit_minuit_gp(num,lnprob):
    minLLH = np.inf
    for i in range(num):
        init0 = np.random.random()*1e9
        init1 = np.random.random()*10000.
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,
                    error_Amp = 10, error_length = 0.1,
                    limit_Amp = (100,1e15), limit_length = (4000,20000000))
        
        m.migrad()

        if m.fval < minLLH:
            #print(m.migrad_ok())
            m_best = m
            minLLH = m.fval
            best_fit_parameters = m.args
            best_fit_parameters_errors = m.np_errors()
            
    print("min LL",minLLH)
    print("best fit parameters",best_fit_parameters)
    return minLLH, best_fit_parameters, best_fit_parameters_errors

class log_like_gp_matern:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __call__(self,Amp,length):
        kernel = Amp*george.kernels.Matern32Kernel(metric=length)
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf


tf = r.TFile.Open("diphox_shape_withGJJJDY_WithEffCor.root")

tf.cd()
tf.ReadAll()

h_hist = tf.Get("Mgg_CP0").Clone()
binwidth = h_hist.GetBinWidth(1)
nbins = h_hist.GetNbinsX()
xmin = h_hist.GetXaxis().GetXmin()
xmax = h_hist.GetXaxis().GetXmax()
h_hist.Rebin(int(2./((xmax-xmin)/nbins)))
h_hist.Scale(10)

Ntoys = 1
mean = 125.
sigma = 2
Amp = 2000

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(Amp,mean,sigma)
signal.SetParNames("Amplitude","mean","Sigma")

Bern5 = r.TF1("Bern5",Bern,xmin,xmax,8)
Bern5.SetParameters(1,0.1,0.01,0.001,0.0001,0.00001)
Bern5.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
Bern5.FixParameter(6,xmin)
Bern5.FixParameter(7,xmax)
h_hist.Fit(Bern5,"SR0")

mass = np.zeros(h_hist.GetNbinsX())
toy = np.zeros(h_hist.GetNbinsX())
Bkg = np.zeros(h_hist.GetNbinsX())
signal_dist = np.zeros(h_hist.GetNbinsX())
lum = np.array([1,10,100])


for i in range(1,h_hist.GetNbinsX()+1):
    mass[i-1] = h_hist.GetBinCenter(i)
    Bkg[i-1] = Bern5(mass[i-1])
    signal_dist[i-1] = signal(mass[i-1])

res_SE = np.zeros(len(mass))
res_Matern = np.zeros(len(mass))

for l in lum:
    for t in range(Ntoys):
        toy = l*(Bkg)
        toy = l*(Bkg+signal_dist)
        #for i in range(len(mass)):
        #    toy[i] = R.Poisson(l*Bkg[i])

        lnprob = log_like_gp(mass,toy)
        minimumLLH, best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp(100,lnprob)
        print("Amp",best_fit_parameters[0],"+/-",best_fit_parameters_errors[0])
        print("Length",best_fit_parameters[1],"+/-",best_fit_parameters_errors[1])
        kernel = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        gp = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        gp.compute(mass,yerr=np.sqrt(toy))
        covarmatrix = gp.get_matrix(mass)
        invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
        covarprod = np.matmul(covarmatrix,invcovarmatrix)
        y_pred_SE, y_var_se = gp.predict(toy,mass,return_var=True)
        Effndf = covarprod.trace()

        chi2 = np.sum((toy-y_pred_SE)**2/y_pred_SE)
        chi2_ndf = chi2/(len(toy) - Effndf)
        print("Chi2/ndf SE",chi2_ndf)


        res_SE += toy-y_pred_SE
        """
        lnprob = log_like_gp_matern(mass,toy)
        minimumLLH,best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp(100,lnprob)
        print("Amp",best_fit_parameters[0],"+/-",best_fit_parameters_errors[0])
        print("Length",best_fit_parameters[1],"+/-",best_fit_parameters_errors[1])
        kernel = best_fit_parameters[0]*george.kernels.Matern32Kernel(metric=best_fit_parameters[1])
        gp = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        gp.compute(mass,yerr=np.sqrt(toy))
        covarmatrix = gp.get_matrix(mass)
        invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
        covarprod = np.matmul(covarmatrix,invcovarmatrix)
        y_pred_m, y_var_m = gp.predict(toy,mass,return_var=True)
        Effndf = covarprod.trace()

        chi2 = np.sum((toy-y_pred_m)**2/y_pred_m)
        chi2_ndf = chi2/(len(toy) - Effndf)
        print("Chi2/ndf Matern",chi2_ndf)
        res_Matern += toy-y_pred_m
        """
    
    plt.figure(1)
    plt.scatter(mass,toy,marker='.',color='k')
    plt.plot(mass,y_pred_SE,'r-',label='GP SE')
    #plt.plot(mass,y_pred_m,'b-',label='GP Matern')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Events")
    plt.title('Distribution Lum %.0f'%(36*l))

    plt.figure(2)
    plt.plot(mass,np.zeros(len(mass)),color='k')
    plt.plot(mass,res_SE,'r-',label='Res GP SE')
    #plt.plot(mass,res_Matern,'b-',label='Res GP Matern')
    
    #plt.fill_between(mass,toy-y_pred-np.sqrt(y_var),toy-y_pred+np.sqrt(y_var),color='k',alpha=0.5,label=r'$1\sigma$')
    #plt.plot(mass,l*signal_dist,'c-.')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title('SE vs Matern Lum %.0f'%(36*l))
    plt.show()

