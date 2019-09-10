import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.optimize import minimize
from  scipy.interpolate import interp1d
import scipy.stats as st
import george

R = r.TRandom(0)

SignalOn = True
SignalOn = False

sigmodel = True
sigmodel = False

Blind = True
Blind = False


def background_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))
    
def signal_function(vars, pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)

def sig_plus_bgr(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100)) + pars[6]/(np.sqrt(2*np.pi)*pars[5])*np.exp(-0.5*((vars[0]-pars[4])/pars[5])**2)

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

Ntoys = 100
mean = 135
sigma = 2
Amp = 200

if sigmodel:
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

#h_chi2_ge = r.TH1D("h_chi2_ge","",200000,0,1000000)
#h_chi2_param = h_chi2_ge.Clone("h_chi2_param")
h_chi2_ge = np.zeros(Ntoys)
h_chi2_param = np.zeros(Ntoys)

chi2_lum_ge = np.zeros(len(lum))
chi2_lum_ge_err = np.zeros(len(lum))
chi2_lum_par = np.zeros(len(lum))
chi2_lum_par_err = np.zeros(len(lum))

mass = np.zeros(h_hist.GetNbinsX())
toy = np.zeros(h_hist.GetNbinsX())
truth = np.zeros(h_hist.GetNbinsX())
#h_hist.Draw("pe")
Error = 0
index = 0

canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
canvas1.SetLeftMargin(0.125)
canvas1.SetBottomMargin(0.125)

""" Blind signal region
if Blind:
    for i_bin in range(nbins):
        if h_hist.GetBinCenter(i_bin) >= mean - 5 and h_hist.GetBinCenter(i_bin) <= mean + 5:
            h_hist.SetBinContent(i_bin,0)
            h_hist.SetBinError(i_bin,0)
"""

for l in lum:
    for t in range(Ntoys):
        for i_bin in range(1,h_hist.GetNbinsX()+1):
            mass[i_bin-1] = h_hist.GetBinCenter(i_bin)
            if SignalOn:
                toy[i_bin-1] = R.Poisson(l*(h_hist.GetBinContent(i_bin) + signal(mass[i_bin-1])))
            else:
                toy[i_bin-1] = R.Poisson(l*h_hist.GetBinContent(i_bin))
            h_toy.SetBinContent(i_bin,toy[i_bin-1])


        if Blind:
            toy = np.delete(toy,[mean-2,mean-1,mean,mean+1,mean+2])
            print(toy)
            h_toy.SetBinContent(i_bin,0)
        print(toy)
        
        #canvas1.Update()
        #input("Enter!")
        #h_toy.Draw("pe")
        #canvas1.Update()
        #input("Enter!")

        """ Ad-hoc function"""

        if sigmodel:
            fit_function.SetParameters(1,1,-0.01,0,mean,sigma,Amp)
            fit_function.FixParameter(0,1)
        else:
            fit_function.SetParameters(1,1,-0.01,0)
            fit_function.FixParameter(0,1)
        
        
        fitresults = h_toy.Fit(fit_function,"SRW")
        
        if fitresults.Status() != 0:
            Error += 1
        print(t+1)

        #h_chi2_param.Fill(fitresults.Chi2()/fitresults.Ndf())
        h_chi2_param[t] = fitresults.Chi2()/fitresults.Ndf()

        """ George"""

        kernel_ge = np.median(toy)*george.kernels.ExpSquaredKernel(metric=1.0)#,block=(1000,100000000))
        ge = george.GP(kernel_ge,solver=george.HODLRSolver)
        ge.compute(mass,yerr=np.sqrt(toy))
        m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)
        ge.set_parameter_vector(m.x)
        print(ge.get_parameter_vector())
        y_pred,y_cov = ge.predict(toy,mass)
        chi2_ge = np.sum((toy-y_pred)**2/toy)
        #h_chi2_ge.Fill(chi2_ge/(len(toy) - 1 - len(ge.get_parameter_vector())))
        h_chi2_ge[t] = chi2_ge/(len(toy) - 1 - len(ge.get_parameter_vector()))
        print("George Chi2/ndf",h_chi2_ge[t],"Ad-hoc Chi2/ndf",h_chi2_param[t])

        h_toy.Draw("pe")
        canvas1.Update()
        
        plt.clf()
        plt.scatter(mass,toy,c='r',alpha=0.8)
        plt.plot(mass,y_pred,'b-')
        plt.pause(0.05)

    #chi2_lum_ge[index] = h_chi2_ge.GetMean()
    #chi2_lum_ge_err[index] = h_chi2_ge.GetMeanError()
    #chi2_lum_par[index] = h_chi2_param.GetMean()
    #chi2_lum_par_err[index] = h_chi2_param.GetMeanError()
    chi2_lum_ge[index] = np.mean(h_chi2_ge)
    chi2_lum_par[index] = np.mean(h_chi2_param)
    index += 1

plt.figure(2)
plt.errorbar(lum,chi2_lum_ge,yerr=chi2_lum_ge_err,marker="o",label='GP George, average: %3f' % np.mean(chi2_lum_ge))
#plt.errorbar(lum,chi2_lum_par,yerr=chi2_lum_par_err,marker="o",label='Ad-hoc, average: %3f' % np.mean(chi2_lum_par))
plt.xlabel("Luminosity scale factor")
plt.ylabel(r'$\chi^2$/ndf')
plt.legend()
plt.show()
#input("Press enter to exit!")


