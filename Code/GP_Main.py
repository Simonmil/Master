import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.optimize import minimize
from scipy.interpolate import BPoly
import george
import sys

R = r.TRandom(0)

SignalOn = True
SignalOn = False

Sigmodel = True
Sigmodel = False

def background_function(vars,pars):
    print(type(vars),type(pars))
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0]-100)*(vars[0]-100))

def signal_function(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)

def sig_plus_bgr(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100)) + pars[6]/(np.sqrt(2*np.pi)*pars[5])*np.exp(-0.5*((vars[0]-pars[4])/pars[5])**2)

#def Bernstein_poly(vars,pars):
#    return 

#Bernstein_poly()
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
nbins = h_hist.GetNbinsX()
binwidth = h_hist.GetBinWidth(1)

Ntoys = int(sys.argv[1])
mean = 135
sigma = 2
Amp = 200

if Sigmodel:
    fit_function = r.TF1("fit_function",sig_plus_bgr,xmin,xmax,7)
    fit_function.SetParameters(1,2,-0.03,1e-9,mean,sigma,Amp)
    fit_function.SetParNames("Norm","a","b","c","Mean","Sigma","Amplitude")
else:
    fit_function = r.TF1("fit_function",background_function,xmin,xmax,4)
    fit_function.SetParameters(1,2,-0.03,1e-9)
    fit_function.SetParNames("Norm","a","b","c")


signal = r.TF1("signal",signal_function,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")
h_toy = h_hist.Clone("h_toy")
h_toy.Reset()


lum = np.array([1,2,5,10,15,20,25,40,50,60,80,100])

h_chi2_ge = np.zeros(Ntoys)
h_chi2_param = np.zeros(Ntoys)

chi2_lum_ge = np.zeros(len(lum))
chi2_lum_ge_err = np.zeros(len(lum))
chi2_lum_par = np.zeros(len(lum))
chi2_lum_par_err = np.zeros(len(lum))

mass = np.zeros(h_hist.GetNbinsX())
toy = np.zeros(h_hist.GetNbinsX())
truth = np.zeros(h_hist.GetNbinsX())
Error = 0
index = 0

canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
canvas1.SetLeftMargin(0.125)
canvas1.SetBottomMargin(0.125)

