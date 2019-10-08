import numpy as np
import matplotlib.pyplot as plt
import george
import ROOT as r
from scipy.interpolate import BPoly
from iminuit import Minuit

R = r.TRandom(0)


def epoly2(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0] - 100)*(vars[0] - 100))


def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])


def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0] - pars[0])/pars[1])**2)


class log_like_gp:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __call__(self,Amp,length):
        kernel = Amp * george.kernels.ExpSquaredKernel(metric=length)
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)

        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf

def fit_minuit_gp(num,lnprob):
    minLLH = np.inf
    best_fit_parameters = (0,0)
    for i in range(num):
        print(i+1)
        init0 = np.random.random()*1e2
        init1 = np.random.random()*10.
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,
                    error_Amp = 10, error_length = 0.1,
                    limit_Amp = (100,1e15), limit_length = (1,1000))
        
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



Ntoys = 10
mean = 125
sigma = 2
Amp = 200


signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")

fit_function = r.TF1("fit_function",epoly2,xmin,xmax,4)
fit_function.SetParameters(1,1,-0.01,0)
fit_function.SetParNames("Norm","a","b","c")

h_toy = h_truth.Clone("h_toy")
h_toy.Reset()
lum = np.array([1,50,100,500,1000])
lum = np.array([1,15,30,50,70,85,100])


mass = np.zeros(h_truth.GetNbinsX())
toy = np.zeros(h_truth.GetNbinsX())
truth = np.zeros(h_truth.GetNbinsX())


"""
Here I fit the GP to a toy, to set the optimal hyperparameters
"""
"""
params = np.zeros((10,2))
chi2_fit = np.zeros(10)
minLLH = np.zeros(10)
bestminLHH = np.inf
"""


for i_bin in range(1,h_truth.GetNbinsX()+1):
    mass[i_bin-1] = h_truth.GetBinCenter(i_bin)
    toy[i_bin-1] = R.Poisson(Bern5(mass[i_bin-1]))
#lnprob = log_like_gp(mass,toy)
#minLLH, best_fit_parameters = fit_minuit_gp(100,lnprob)
#params[i][:] = fit_parameters



"""
kernel = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
gp = george.GP(kernel,solver=george.HODLRSolver)
gp.compute(mass,yerr=np.sqrt(toy))
y_pred = gp.predict(toy,mass)[0]

chi2 = np.sum((toy-y_pred)**2/y_pred)
chi2_fit = chi2/(len(toy)-len(gp.get_parameter_vector()))
"""
#if minLLH[i] < bestminLHH:
#    bestminLHH = minLLH[i]
#    best_fit_parameters = fit_parameters


#print("Min LLH",minLLH,"Params",np.log(best_fit_parameters[0]),np.log(best_fit_parameters[1]))


chi2_mean_gp = np.zeros(len(lum))
chi2_mean_par = np.zeros(len(lum))
chi2_fit = np.zeros(Ntoys)
h_chi2 = r.TH1D("h_chi2","Chi2 ad-hoc",100,0,20)
color = ['r','b','g','c','m','k','chartreuse']
index = 0
Error = 0

canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
canvas1.SetLeftMargin(0.125)
canvas1.SetBottomMargin(0.125)

hs = r.THStack("hs","Chi2 ad-hoc")
plt.figure(1)
for l in lum:
    print("Luminosity: ",l)
    for i in range(Ntoys):
        for i_bin in range(1,h_truth.GetNbinsX()+1):
            mass[i_bin-1] = h_truth.GetBinCenter(i_bin)
            toy[i_bin-1] = R.Poisson(l*Bern5(mass[i_bin-1]))
            h_toy.SetBinContent(i_bin,toy[i_bin-1])
            h_toy.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))

        """ad-hoc"""
        fit_function.SetParameters(1,1,-0.01,1e-6)
        fit_function.FixParameter(0,1)       

        fitresults = h_toy.Fit(fit_function,"SPR0Q")
        canvas1.Update()
        if fitresults.Status() != 0:
            Error += 1

        h_chi2.Fill(fitresults.Chi2()/fitresults.Ndf())


        """george"""
        if i%1000 == 0:
            print(i/1000.)
        
        lnprob = log_like_gp(mass,toy)
        minLLH, best_fit_parameters = fit_minuit_gp(100,lnprob)
        kernel = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        #kernel = np.exp(100)*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        gp = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        gp.compute(mass,yerr=np.sqrt(toy))

        y_pred = gp.predict(toy,mass)[0]

        chi2 = np.sum((toy-y_pred)**2/y_pred)
        chi2_fit[i] = chi2/(len(toy)-len(gp.get_parameter_vector()))

        #plt.clf()
        #plt.scatter(mass,toy,c='r',alpha=0.8)
        #plt.plot(mass,y_pred,'b-')
        #plt.pause(0.05)

    chi2_mean_par[index] = h_chi2.GetMean()
    h_chi2_l = h_chi2.Clone("h_chi2_l")
    hs.Add(h_chi2_l)

    h_chi2.Reset()
    plt.hist(chi2_fit,bins=50,color=color[index],label='Lum: %.1f'%l,alpha=0.8,histtype='step')
    plt.xlabel("Chi2")
    plt.ylabel("#")
    chi2_mean_gp[index] = np.mean(chi2_fit)

    index += 1

hs.Draw("plc nostack")
canvas1.Update()
plt.legend()
plt.title("Test statistic distribution")
#plt.show()

plt.figure(2)

plt.plot(lum,chi2_mean_gp,color='r', label='GP')
plt.plot(lum,chi2_mean_par,color='b',label='Ad-hoc')
plt.xlabel('Luminosity scale factor')
plt.ylabel(r'$\chi^2/ndf$')
plt.legend()
plt.show()




""" Spurious signal testing"""






