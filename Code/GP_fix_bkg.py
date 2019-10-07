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
    if len(p) > 2:
        p[0] = ge.get_parameter_vector()[0]
        p[1] = ge.get_parameter_vector()[1]
        gesig.set_parameter_vector(p)
        #print("sig:",p)
        return -gesig.log_likelihood(toysig)
    else:
        ge.set_parameter_vector(p)
        #print("bkg:",p)
        return -ge.log_likelihood(toy)

def grad_neg_log_like(p):
    if len(p) > 2:
        p[0] = ge.get_parameter_vector()[0]
        p[1] = ge.get_parameter_vector()[1]
        gesig.set_parameter_vector(p)
        return -gesig.grad_log_likelihood(toysig)
    else:
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
Overfit = 0
Underfit = 0
for i in range(100):
    for i_bin in range(1,h_truth.GetNbinsX()+1):
        mass[i_bin-1] = h_truth.GetBinCenter(i_bin)
        toy[i_bin-1] = R.Poisson(50*Bern5(mass[i_bin-1]))
        toysig[i_bin-1] = toy[i_bin-1] + R.Poisson(50*signal(mass[i_bin-1]))

    kernel_bkg = np.median(toy)*george.kernels.ExpSquaredKernel(metric=np.exp(6))
    ge = george.GP(kernel_bkg,solver=george.HODLRSolver,mean=np.median(toy))
    ge.compute(mass,yerr=np.sqrt(toy))
    m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)
    ge.set_parameter_vector(m.x)
    y_pred = ge.predict(toy,mass)[0]

    
    chi2 = np.sum((toy-y_pred)**2/y_pred)
    chi2ndf = chi2/(len(toy) - len(ge.get_parameter_vector()))
    print("Chi2 background: ", chi2ndf)

    kernel_sig = 40*george.kernels.LocalGaussianKernel(location=135,log_width=3)
    kernel_bkgsig = kernel_bkg + kernel_sig
    gesig = george.GP(kernel=kernel_bkgsig,solver=george.HODLRSolver)
    gesig.compute(mass,yerr=np.sqrt(toysig))
    m_sig = minimize(neg_log_like,gesig.get_parameter_vector(),jac=grad_neg_log_like,bounds=((20,25),(5,9),(1,100),(130,140),(1,4)))
    gesig.set_parameter_vector(m_sig.x)

    y_pred = gesig.predict(toysig,mass)[0]
    #plt.scatter(mass,toysig,color='r',marker='.')
    #plt.plot(mass,y_pred)
    
    chi2 = np.sum((toysig-y_pred)**2/y_pred)
    #print(gesig.get_parameter_names())
    print(chi2)
    chi2ndf = chi2/(len(toysig) - len(gesig.get_parameter_vector()))
    print("Chi2 signal + background: ", chi2ndf)
    if chi2ndf<0.1:
        Overfit += 1
    if chi2ndf>10:
        Underfit += 1    
    print(gesig.get_parameter_vector())
    plt.clf()
    plt.scatter(mass,toysig,color='r',marker='.')
    plt.plot(mass,y_pred)
    plt.pause(0.05)

print(Overfit)
print(Underfit)