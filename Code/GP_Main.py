import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import BPoly
import george
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from iminuit import Minuit
import time

R = r.TRandom(0)

SignalOn = False
SignalOn = True

Sigmodel = True
Sigmodel = False

Epoly2_fit = False
Epoly2_fit = True


def epoly2(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0]-100)*(vars[0]-100))


def Gaussian(vars,pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)


def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])
    #bp = BPoly(pars[0:-2],pars[-2:])
    #return bp(vars[0])


class log_like_gp:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __call__(self,Amp,length):
        kernel = Amp * george.kernels.ExpSquaredKernel(metric=length)
        gp = george.GP(kernel = kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y,self.x)
        except:
            np.inf

def fit_minuit_gp(num,lnprob):
    minLLH = np.inf
    best_fit_parameters = (0,0)
    for i in range(num):
        #print(i+1)
        init0 = np.random.random()*1e2
        init1 = np.random.random()*10.
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,
                    error_Amp = 10,error_length = 0.1,
                    limit_Amp = (100.,1e15), limit_length = (1,20000))
        
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_parameters = m.args
    
    print("min LL",minLLH)
    print("best fit parameters",best_fit_parameters)
    return minLLH, best_fit_parameters

class log_like_gp_sig:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __call__(self,Amp,length,Sigamp,sigma,mass):
        
        kernel1 = Amp * george.kernels.ExpSquaredKernel(metric=length)
        kernel2 = Sigamp*george.kernels.LocalGaussianKernel(location=mass,log_width=sigma)
        kernel = kernel1 + kernel2
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
        #print(gp.get_parameter_vector())
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf

def fit_minuit_gp_sig(num,lnprob):
    minLLH = np.inf
    best_fit_parameters = (0,0)
    for i in range(num):
        #print(i+1)
        init0 = np.random.random()*1e9
        init1 = np.random.random()*10000.
        #init2 = np.random.random()*10.
        #init3 = np.random.random()*1.
        #init4 = np.random.random()*10.
        init2 = 0
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,Sigamp=init2,sigma=1,mass=125,
                    error_Amp = 1000, error_length = 10,error_Sigamp=1,error_sigma=0.01,error_mass=5,
                    limit_Amp = (100,1e15), limit_length = (4000,20000), limit_Sigamp=(0,10000000000000),limit_sigma=(0,4),limit_mass=(100,180),
                    fix_sigma = False,fix_mass = True)
        
        m.migrad()
        #print(m.migrad_ok())
        if m.fval < minLLH:
            #print(m.np_errors())
            print(m.migrad_ok())
            #print(m.args[2]/m.np_errors()[2])
            m_best = m
            minLLH = m.fval
            best_fit_parameters = m.args
            best_fit_parameters_errors = m.np_errors()
            #print(m.mnprofile("Sigamp",bound=(best_fit_parameters[2]/2,best_fit_parameters[2]*1.5),bins=3))
            
    #m_best.minos()
    #m_best.draw_mnprofile("Sigamp")
    print("min LL",minLLH)
    print("best fit parameters",best_fit_parameters)
    return minLLH, best_fit_parameters, best_fit_parameters_errors

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
""" Now we have read the file containing the histogram and rebinned for H->gamgam. To remove any poisson noise as much as possible
we fit the data with a Bernstein polynomial of degreee 5. This polynomial, with added Poisson noise, epoly2 should be able to fit
for small luminosities. Then we see if epoly2 and GP can follow for increasing Lum. We will also fit with other polynomials, like Bern 3 and 4."""

Bern5 = r.TF1("Bern5",Bern,xmin,xmax,8)
Bern5.SetParameters(1,0.1,0.01,0.001,0.0001,0.00001)
Bern5.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
Bern5.FixParameter(6,xmin)
Bern5.FixParameter(7,xmax)
h_hist.Fit(Bern5,"SR0")
Bern5_dist = np.zeros(h_hist.GetNbinsX())
signal_dist = np.zeros(h_hist.GetNbinsX())
mass = np.zeros(h_hist.GetNbinsX())


"""
The fitting of toy models.
Here, ROOT is used to fit the toy distributions with an ad-hoc function, and GP is fitted using George.
"""

Ntoys = 1
mean = 125
sigma = 2
Amp = 2000



fit_function = r.TF1("fit_function",epoly2,xmin,xmax,4)
fit_function.SetParameters(1,1,-0.01,0)
fit_function.SetParNames("Norm","a","b","c")

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")
h_toy = h_hist.Clone("h_toy")
h_toy.Reset()
lum = np.array([1,15,30,50,60,80,100])
#lum = np.array([1,125,500,750,1000,2500,5000,7500,10000,12500])
lum = np.array([10])
h_chi2_ge = np.zeros(Ntoys)
h_chi2_param = np.zeros(Ntoys)


chi2_mean_ge = np.zeros(len(lum))
chi2_mean_ge_err = np.zeros(len(lum))
chi2_mean_par = np.zeros(len(lum))
chi2_mean_par_err = np.zeros(len(lum))

h_mean_best_Amplitude = np.zeros(len(lum))
h_mean_best_lengthscale = np.zeros(len(lum))
h_best_Amplitude = np.zeros(Ntoys)
h_best_lengthscale = np.zeros(Ntoys)

toy = np.zeros(h_hist.GetNbinsX())
truth = np.zeros(h_hist.GetNbinsX())
Error = 0
index = 0
Overfit = 0
res = np.zeros(h_hist.GetNbinsX())
fitfunction = np.zeros(h_hist.GetNbinsX())


h_chi2 = r.TH1D("h_chi2","Chi2 ad-hoc",100,0,20)
color = ['r','b','g','c','m','k','chartreuse','lime']


#canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
#canvas1.SetLeftMargin(0.125)
#canvas1.SetBottomMargin(0.125)

hs = r.THStack("hs","Chi2 ad-hoc")

for l in lum:
    for i in range(1,h_hist.GetNbinsX()+1):
        mass[i-1] = h_hist.GetBinCenter(i)
        Bern5_dist[i-1] = l*Bern5(mass[i-1])
        signal_dist[i-1] = l*signal(mass[i-1])
    for t in range(Ntoys):
        print(t+1)
        for i_bin in range(1,h_hist.GetNbinsX()+1):
            if SignalOn:
                toy[i_bin-1] = R.Poisson(Bern5_dist[i_bin-1] + signal_dist[i_bin-1])
                #toy[i_bin-1] = Bern5_dist[i_bin-1] + signal_dist[i_bin-1]
            else:
                #toy[i_bin-1] = Bern5_dist[i_bin-1]
                toy[i_bin-1] = R.Poisson(Bern5_dist[i_bin-1])
            h_toy.SetBinContent(i_bin,toy[i_bin-1]) 
            h_toy.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))
        
        if Epoly2_fit:
            fit_function.SetParameters(1,1,-0.01,1e-6)
            fit_function.FixParameter(0,1)
            fitresults = h_toy.Fit(fit_function,"SPR0Q")
            #fit_params = fitresults.GetParameters()
            #fit_params_err = fitresults.GetErrors()
            for i in range(1,h_hist.GetNbinsX()+1):
                fitfunction[i-1] = fit_function(mass[i-1])
            if fitresults.Status() != 0:
                Error += 1
            h_chi2_par = fitresults.Chi2()/fitresults.Ndf()
            h_chi2.Fill(h_chi2_par)
        
        print(h_toy.Integral())
        """George"""
             
        lnprob = log_like_gp(mass,toy)
        minimumLLH, best_fit_params = fit_minuit_gp(100,lnprob)
        kernel_ge = best_fit_params[0]*george.kernels.ExpSquaredKernel(metric=best_fit_params[1])
        ge = george.GP(kernel_ge,solver=george.HODLRSolver,mean=np.median(toy))
        ge.compute(mass,yerr=np.sqrt(toy))
        #print(ge.get_parameter_vector())
        y_pred, y_var = ge.predict(toy,mass,return_var=True)
        print(np.sqrt(best_fit_params[0]))
        
        lnprob_sig = log_like_gp_sig(mass,toy)
        minLLH_sig,best_fit_parameters_sig,best_fit_parameters_sig_errors = fit_minuit_gp_sig(100,lnprob_sig)
        kernel1 = best_fit_parameters_sig[0] * george.kernels.ExpSquaredKernel(metric=best_fit_parameters_sig[1])
        kernel2 = best_fit_parameters_sig[2]*george.kernels.LocalGaussianKernel(location=best_fit_parameters_sig[4],log_width=best_fit_parameters_sig[3])
        kernel = kernel1 + kernel2
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver,mean=np.median(toy))
        gp.compute(mass,yerr=np.sqrt(toy))

        Sigamp = np.sqrt(best_fit_parameters_sig[2])
        Sigamperror = best_fit_parameters_sig_errors[2]
        print(Sigamp,r'$\pm$',Sigamperror/(2*Sigamp))
        y_pred_sig, y_covar_sig = gp.predict(toy,mass,return_var=False)
        
        #plt.contourf(mass,mass,y_covar_sig)
        #plt.colorbar()
        #plt.show()
        #foobar
        #h_best_Amplitude[t] = best_fit_params[0]
        #h_best_lengthscale[t] = best_fit_params[1]
        
        y_pred, y_var = gp.predict(toy,mass,return_var = True)
        chi2_ge = np.sum((toy-y_pred)**2/y_pred)
        h_chi2_ge[t] = chi2_ge/(len(toy) - len(ge.get_parameter_vector()))
        print("George Chi2/ndf",h_chi2_ge[t])#,"Ad-hoc Chi2/ndf",h_chi2_par)

        if t%1000 == 0:
            print(t/1000.)        
         
        if h_chi2_ge[t] < 0.01:
            Overfit += 1
        
        #mass_110135 = []
        #Bkg_110135 = []
        #y_pred_110135 = []
        #y_var_110135 = []
        #signal_dist_110135 = []
        #for i in range(len(mass)):
        #    if mass[i-1] >= 110 and mass[i-1] <= 135:
        #        mass_110135.append(mass[i-1])
        #        Bkg_110135.append(Bern5_dist[i-1])
        #        y_pred_110135.append(y_pred[i-1])
        #        y_var_110135.append(y_var[i-1])
        #        signal_dist_110135.append(signal_dist[i-1])
        #mass_110135 = np.array(mass_110135)
        #Bkg_110135 = np.array(Bkg_110135)
        #y_pred_110135 = np.array(y_pred_110135)
        #y_var_110135 = np.array(y_var_110135)
        #signal_dist_110135 = np.array(signal_dist_110135)


        plt.clf()
        plt.plot(mass,y_pred,'b-')
        plt.scatter(mass,toy,color='r')
        #plt.fill_between(mass,y_pred-y_var,y_pred+y_var,color='g',alpha=0.5)
        #plt.plot(mass,Bern5_dist-Bern5_dist,color='r')
        #plt.scatter(mass,toy-Bern5_dist,c='k',marker='.')
        #plt.plot(mass,y_pred-Bern5_dist,'b-',label='Res GP')
        #plt.plot(mass,fitfunction-Bern5_dist,'g-',label='Res ad-hoc')
        #plt.plot(mass,signal_dist,'m-.',label='signal')
        #plt.fill_between(mass,y_pred-Bern5_dist-y_var/100.,y_pred-Bern5_dist+y_var/100.,color='k',alpha=0.5,label='1% variance')
        #plt.scatter(mass_110135,Bkg_110135-Bkg_110135,c='k',marker='.')
        #plt.plot(mass_110135,y_pred_110135-Bkg_110135,'b-')
        #plt.fill_between(mass_110135,y_pred_110135-Bkg_110135-y_var_110135,y_pred_110135-Bkg_110135+y_var_110135,color='g',alpha=0.5,label='variance')
        plt.legend()
        plt.xlabel(r"$m_{\gamma\gamma}[GeV]$")
        plt.ylabel("Residuals")
        plt.title(r'Luminosity $\int Ldt = %.1f fb^{-1}$'%(36*l))
        plt.pause(0.01)
        #plt.show()
        
    h_mean_best_Amplitude[index] = np.mean(h_best_Amplitude)
    h_mean_best_lengthscale[index] = np.mean(h_best_lengthscale)
    
    chi2_mean_ge[index] = np.mean(h_chi2_ge)
    chi2_mean_ge_err[index] = np.std(h_chi2_ge)
    chi2_mean_par[index] = h_chi2.GetMean()
    chi2_mean_par_err[index] = h_chi2.GetStdDev()
    h_chi2_l = h_chi2.Clone("h_chi2_l")
    hs.Add(h_chi2_l)

    h_chi2.Reset()
    #plt.figure(2)
    #plt.hist(h_chi2_ge,bins=50,color=color[index],label='Lum: %.1f'%l,alpha=0.8,histtype='step')
    #plt.xlabel("Chi2")
    #plt.ylabel("#")

    index += 1


#plt.figure(1)

#hs.Draw("plc nostack")
#canvas1.Update()
#plt.legend()
#plt.title("Test statistic distribution")


plt.figure(2)
plt.errorbar(lum,chi2_mean_ge,yerr=chi2_mean_ge_err,marker=".",label='GP',c='b')
plt.errorbar(lum,chi2_mean_par,yerr=chi2_mean_par_err,marker=".",label='Ad hoc',c='r')

plt.xlabel("Lum scale")
plt.ylabel(r'$\chi^2$/ndf')
plt.legend()
plt.title("Chi2/ndf Ad-hoc and GP")


#plt.figure(3)
#plt.plot(mass,Bern5_dist)




#plt.figure(3)
#plt.plot(lum,chi2_lum_ge,marker=".",label='GP',c='r')
#plt.xlabel("Lum scale")
#plt.ylabel(r'$\chi^2$/ndf')
#plt.legend()
#plt.title("Chi2/ndf GP")

#plt.figure(4)
#plt.plot(lum,h_mean_best_lengthscale,marker='o')
#plt.xlabel("Luminosity scale factor")
#plt.ylabel("Lengthscale")
#plt.figure(5)
#plt.plot(lum,h_mean_best_Amplitude,marker='o',c='b')
#plt.xlabel("Luminosity scale factor")
#plt.ylabel("Amp")


plt.show()

print("Overfitted: ", Overfit)
