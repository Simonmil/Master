import numpy as np
import matplotlib.pyplot as plt
import george
import ROOT as r
from scipy.interpolate import BPoly
from iminuit import Minuit

R = r.TRandom(0)


def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0] - pars[0])/pars[1])**2)


def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])

def model_3params(t,params):
    p0,p1,p2 = params
    sqrts = 13000.
    return (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))


def mean_gp(params,t):
    p0,p1,p2 = params
    sqrts = 13000.
    return (p0 * (1.-t/sqrts)**p1 * (t/sqrts)**(p2))

class log_like_gp:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __call__(self,Amp,length,p0,p1,p2):
        kernel = Amp * george.kernels.ExpSquaredKernel(metric=length)
        gp = george.GP(kernel = kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y - mean_gp((p0,p1,p2),self.x))
        except:
            np.inf

def fit_minuit_gp(num,lnprob):
    minLLH = np.inf
    best_fit_parameters = (0,0,0,0,0)
    for i in range(num):
        print(i+1)
        init0 = np.random.random()*1e2
        init1 = np.random.random()*10.
        init2 = np.random.random()*1.
        init3 = np.random.random()*1.
        init4 = np.random.random()*-1.
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,p0=init2,p1=init3,p2=init4,
                    error_Amp = 10,error_length = 0.1,error_p0=0.001,error_p1=0.001,error_p2=0.001,
                    limit_Amp = (100.,1e15), limit_length = (1,50), limit_p0 = (0,1000), limit_p1 = (0,100), limit_p2 = (-100,100))
        
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_parameters = m.args
    
    print("min LL",minLLH)
    print("best fit parameters",best_fit_parameters)
    return minLLH, best_fit_parameters


tf = r.TFile.Open("diphox_shape_withGJJJDY_WithEffCor.root")

tf.cd()
tf.ReadAll()

h_hist = tf.Get("Mgg_CP0").Clone()
binwidth = h_hist.GetBinWidth(1)
nbins = h_hist.GetNbinsX()
xmin = h_hist.GetXaxis().GetXmin()
xmax = h_hist.GetXaxis().GetXmax()
h_hist.Rebin(int(2./((xmax-xmin)/nbins)))


Bern5 = r.TF1("Bern5",Bern,xmin,xmax,8)
Bern5.SetParameters(1,0.1,0.01,0.001,0.0001,0.00001)
Bern5.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
Bern5.FixParameter(6,xmin)
Bern5.FixParameter(7,xmax)
h_hist.Fit(Bern5,"SR0")
h_truth = Bern5.CreateHistogram()
binwidth = h_truth.GetBinWidth(1)
nbins = h_truth.GetNbinsX()

xmin = h_truth.GetXaxis().GetXmin()
xmax = h_truth.GetXaxis().GetXmax()
h_truth.Rebin(int(2./((xmax-xmin)/nbins))) 

mean = 135
sigma = 2
Amp = 200

mass = np.zeros(h_truth.GetNbinsX())
toy = np.zeros(h_truth.GetNbinsX())
toysig = np.zeros(h_truth.GetNbinsX())
truth = np.zeros(h_truth.GetNbinsX())

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")
h_toy = h_truth.Clone("h_toy")
h_toysig = h_truth.Clone("h_toysig")
h_toy.Reset()
h_toysig.Reset()

lum = [1,25,50,75,100]
for l in lum:
    for t in range(5):
        for i_bin in range(1,h_truth.GetNbinsX()+1):
            mass[i_bin-1] = h_truth.GetBinCenter(i_bin)
            toy[i_bin-1] = R.Poisson(l*Bern5(mass[i_bin-1]))

        lnprob = log_like_gp(mass,toy)
        minimumLLH, BFP = fit_minuit_gp(100,lnprob)
        kernel = BFP[0]*george.kernels.ExpSquaredKernel(metric=BFP[1])
        gp = george.GP(kernel=kernel)
        gp.compute(mass,yerr=np.sqrt(toy))
        par = np.zeros(len(BFP)-2)
        par[0] = BFP[2]
        par[1] = BFP[3]
        par[2] = BFP[4]

        meanGP,var_gp = gp.predict(toy - mean_gp(par,mass),mass,return_var=True)

        meanGPnom = meanGP + model_3params(mass,par)

        chi2 = np.sum((toy-meanGPnom)**2/meanGPnom)
        chi2_ndf = chi2/(len(toy)-len(gp.get_parameter_vector()))
        print(chi2_ndf)

        plt.clf()
        plt.scatter(mass,toy,color='r',marker='.')
        plt.plot(mass,meanGPnom)
        plt.fill_between(mass,meanGPnom-np.sqrt(var_gp),meanGPnom+np.sqrt(var_gp),color='g',alpha=0.2)
        plt.pause(0.05)

