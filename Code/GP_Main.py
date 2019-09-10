import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.optimize import minimize
from scipy.interpolate import BPoly
import george
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



R = r.TRandom(0)

SignalOn = True
SignalOn = False

Sigmodel = True
Sigmodel = False


def epoly2(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0]-100)*(vars[0]-100))


def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)

def sig_plus_bgr(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100)) + pars[6]/(np.sqrt(2*np.pi)*pars[5])*np.exp(-0.5*((vars[0]-pars[4])/pars[5])**2)

def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])
    #bp = BPoly(pars[0:-2],pars[-2:])
    #return bp(vars[0])


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





""" Now we have read the file containing the histogram and rebinned for H->gamgam. To remove any poisson noise as much as possible
we fit the data with a Bernstein polynomial of degreee 5. This polynomial, with added Poisson noise, epoly2 should be able to fit
for small luminosities. Then we see if epoly2 and GP can follow for increasing Lum. We will also fit with other polynomials, like Bern 3 and 4."""

Bern5 = r.TF1("Bern5",Bern,xmin,xmax,8)
Bern5.SetParameters(1,0.1,0.01,0.001,0.0001,0.00001)
Bern5.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
Bern5.FixParameter(6,xmin)
Bern5.FixParameter(7,xmax)
h_hist.Fit(Bern5,"SRW0Q")
h_truth = Bern5.CreateHistogram()
binwidth = h_truth.GetBinWidth(1)
nbins = h_truth.GetNbinsX()
xmin = h_truth.GetXaxis().GetXmin()
xmax = h_truth.GetXaxis().GetXmax()
h_truth.Rebin(int(2./((xmax-xmin)/nbins)))



"""
The fitting of toy models.
Here, ROOT is used to fit the toy distributions with an ad-hoc function, and GP is fitted using George.
"""

Ntoys = int(sys.argv[1])
mean = 135
sigma = 2
Amp = 200

if Sigmodel:
    fit_function = r.TF1("fit_function",sig_plus_bgr,xmin,xmax,7)
    fit_function.SetParameters(1,1,-0.01,0,mean,sigma,Amp)
    fit_function.SetParNames("Norm","a","b","c","Mean","Sigma","Amplitude")
else:
    fit_function = r.TF1("fit_function",epoly2,xmin,xmax,4)
    fit_function.SetParameters(1,1,-0.01,0)
    fit_function.SetParNames("Norm","a","b","c")

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")
h_toy = h_truth.Clone("h_toy")
h_toy.Reset()
lum = np.array([1,2,5,10,15,20,25,40,50,60,80,100])

h_chi2_ge = np.zeros(Ntoys)
h_chi2_param = np.zeros(Ntoys)
h_chi2_sk = np.zeros(Ntoys)

chi2_lum_ge = np.zeros(len(lum))
chi2_lum_ge_err = np.zeros(len(lum))
chi2_lum_par = np.zeros(len(lum))
chi2_lum_par_err = np.zeros(len(lum))
chi2_lum_sk = np.zeros(len(lum))

mass = np.zeros(h_truth.GetNbinsX())
toy = np.zeros(h_truth.GetNbinsX())
truth = np.zeros(h_truth.GetNbinsX())
Error = 0
index = 0

#canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
#canvas1.SetLeftMargin(0.125)
#canvas1.SetBottomMargin(0.125)

for l in lum:
    for t in range(Ntoys):
        print(t+1)
        for i_bin in range(1,h_truth.GetNbinsX()+1):
            mass[i_bin-1] = h_truth.GetBinCenter(i_bin)

            if SignalOn:
                toy[i_bin-1] = R.Poisson(l*(Bern5(mass[i_bin-1]) + signal(mass[i_bin-1])))
            else:
                toy[i_bin-1] = R.Poisson(l*Bern5(mass[i_bin-1]))
            h_toy.SetBinContent(i_bin,toy[i_bin-1])
        
        

        fit_function.SetParameters(1,1,-0.01,1e-6)
        fit_function.FixParameter(0,1)
        
        
        fitresults = h_toy.Fit(fit_function,"SRW0Q")

        
        if fitresults.Status() != 0:
            Error += 1

        h_chi2_param[t] = fitresults.Chi2()/fitresults.Ndf()

        """George"""
        kernel_ge = np.median(toy)*george.kernels.ExpSquaredKernel(metric=1.0)#,block=(100,100000))
        ge = george.GP(kernel_ge,solver=george.HODLRSolver)#,white_noise=np.log(np.sqrt(np.mean(toy))))
        ge.compute(mass,yerr=np.sqrt(toy))
        m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)
        ge.set_parameter_vector(m.x)
        print(ge.get_parameter_vector())
        #print(ge.get_parameter_bounds())
        #print(ge.get_parameter_dict())
        y_pred,y_cov = ge.predict(toy,mass)
        chi2_ge = np.sum((toy-y_pred)**2/toy)
        h_chi2_ge[t] = chi2_ge/(len(toy) - 1 - len(ge.get_parameter_vector()))
        #print(ge.get_parameter_names())
        #print(ge.get_parameter_vector())
        #print(ge.get_parameter_bounds())
        
        """sklearn
        kernel_sk = np.mean(toy)*RBF(length_scale=1.0,length_scale_bounds=(0.5,1.5))
        sk = GaussianProcessRegressor(kernel=kernel_sk,alpha=np.sqrt(toy))
        sk.fit(mass.reshape(-1,1),toy)
        y_pred_sk = sk.predict(mass.reshape(-1,1))
        chi2_sk = np.sum((toy - y_pred_sk)**2/toy)
        h_chi2_sk[t] = chi2_sk/(len(toy) - 1 - sk.kernel_.n_dims)
        """

        #print(t+1)
        #print(y_pred)
        print(h_chi2_ge[t])
        plt.clf()
        plt.scatter(mass,toy,c='r',alpha=0.8)
        plt.plot(mass,y_pred,'b-')
        plt.pause(0.05)
        

    chi2_lum_ge[index] = np.mean(h_chi2_ge)
    chi2_lum_par[index] = np.mean(h_chi2_param)
    chi2_lum_sk[index] = np.mean(h_chi2_sk)
    index += 1


#print(np.mean(chi2_lum_ge))
plt.figure(2)
plt.plot(lum,chi2_lum_ge,marker=".",label='GP')
plt.plot(lum,chi2_lum_par,marker=".",label='Ad hoc')
#plt.plot(lum,chi2_lum_sk,marker=".",label='sk')
plt.xlabel("Lum scale")
plt.ylabel(r'$\chi^2$/ndf')
plt.legend()
plt.show()
