import numpy as np
import matplotlib.pyplot as plt
import george
import sys
import ROOT as r
from scipy.optimize import minimize
from scipy.interpolate import BPoly



R = r.TRandom(0)


def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)


def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])


def neg_log_like(p):
    ge.set_parameter_vector(p)
    return -ge.log_likelihood(toy)

def grad_neg_log_like(p):
    ge.set_parameter_vector(p)
    return -ge.grad_log_likelihood(toy)


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


for i_bin in range(1,h_truth.GetNbinsX()+1):
    mass[i_bin-1] = h_truth.GetBinCenter(i_bin)
    toy[i_bin-1] = R.Poisson(Bern5(mass[i_bin-1]))
    toysig[i_bin-1] = R.Poisson(Bern5(mass[i_bin-1]) + signa(mass[i_bin-1]))

    kernel_ge = np.median(toy)*george.kernels.ExpSquaredKernel(metric=np.exp(6))
    ge = george.GP(kernel_ge,solver=george.HODLRSolver)
    ge.compute(mass,yerr=np.sqrt(toy))
    m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)
    ge.set_parameter_vector(m.x)


