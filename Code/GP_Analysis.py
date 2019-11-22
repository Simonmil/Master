import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
import george
import sys
import time
import os
from iminuit import Minuit
from scipy.interpolate import BPoly
from pathlib import Path

R = r.TRandom(0)

Sigplusbkg = True
Sigplusbkg = False


#==================================================
#
# Create folder for saving of results
#
#==================================================

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

save_path = "Figures/" + timestamp + "/"



try:  
    os.makedirs(save_path)
except OSError:  
    print ("Creation of the directory %s failed" % save_path)
    sys.exit(1)
else:  
    print ("Successfully created the directory %s" % save_path)
 
# Create folder for saving of results - end
#==================================================



def epoly2(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0] - 100)*(vars[0] - 100))

def Gaussian(vars,pars):
    return pars[0]/(np.sqrt(2*np.pi)*pars[2])*np.exp(-0.5*((vars[0]-pars[1])/pars[2])**2)

def SigFit(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0] - 100)*(vars[0] - 100)) + pars[6]/(np.sqrt(2*np.pi)*pars[5])*np.exp(-0.5*((vars[0]-pars[4])/pars[5])**2)

def Bern(vars,pars):
    pars_coef = []
    for i in range(len(pars)):
        pars_coef.append(pars[i])
    pars_coef = np.array(pars_coef).reshape(-1,1)
    return BPoly(pars_coef[0:-2],[pars_coef[-2][0],pars_coef[-1][0]])(vars[0])

class log_like_gp_sig:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __call__(self,Amp,length,Sigamp,sigma,mass):
        kernel_bkg = Amp * george.kernels.ExpSquaredKernel(metric=length)
        kernel_sig = Sigamp * george.kernels.LocalGaussianKernel(location=mass,log_width=sigma)
        kernel = kernel_bkg + kernel_sig
        gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
        try:
            gp.compute(self.x,yerr=np.sqrt(self.y))
            return -gp.log_likelihood(self.y)
        except:
            return np.inf

def fit_minuit_gp_sig(num,lnprob,fix=False):
    minLLH = np.inf
    for i in range(num):
        init0 = np.random.random()*1e9
        init1 = np.random.random()*10000.
        init2 = 0
        m = Minuit(lnprob,throw_nan=False,pedantic=False,print_level=0,Amp=init0,length=init1,Sigamp=init2,sigma=2,mass=125,
                    error_Amp = 1000, error_length = 10,error_Sigamp=1,error_sigma=0.01,error_mass=5,
                    limit_Amp = (0,1e15), limit_length = (0,20000), limit_Sigamp=(0,np.exp(30)),limit_sigma=(0,4),limit_mass=(100,180),
                    fix_sigma = fix,fix_mass = fix)
        
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
                    limit_Amp = (100,1e15), limit_length = (4000,20000))
        
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
bkg_plus_sig = np.zeros(h_hist.GetNbinsX())
mass = np.zeros(h_hist.GetNbinsX())
Bkg_fitfunction = np.zeros(h_hist.GetNbinsX())
Sig_fitfunction = np.zeros(h_hist.GetNbinsX())


Ntoys = 1000
mean = 125
sigma = 2*np.sqrt(2)
Amp = 2000
lum = np.array([1,10,100])
#lum = np.array([1,10])

Bkg_fit_function = r.TF1("Bkg_fit_function",epoly2,xmin,xmax,4)
Bkg_fit_function.SetParameters(1,1,-0.01,1e-6)
Bkg_fit_function.SetParNames("Norm","a","b","c")

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(Amp,mean,sigma)
signal.SetParNames("Amplitude","mean","Sigma")

Sig_fit_function = r.TF1("Sig_fit_function",SigFit,xmin,xmax,7)
Bkg_fit_function.SetParameters(1,1,-0.01,1e-6,mean,sigma,Amp)
Bkg_fit_function.SetParNames("Norm","a","b","c","Mean","Sigma","Amplitude")


h_toy = h_hist.Clone("h_toy")
h_toy.Reset()
h_bkg = h_hist.Clone("h_bkg")
h_bkg.Reset()
h_bkgsig = h_hist.Clone("h_bkgsig")
h_bkgsig.Reset()


Effndf = np.zeros(len(lum))

Error = 0
index = 0
Overfit = 0

res = np.zeros(h_hist.GetNbinsX())
fitfunction = np.zeros(h_hist.GetNbinsX())


h_chi2_par = r.TH1D("h_chi2_par","Chi2 ad-hoc",100,0,20)
color = ['r','b','g','c','m','k','chartreuse','lime']


hs = r.THStack("hs","Chi2 ad-hoc")

for i in range(1,h_hist.GetNbinsX()+1):
    mass[i-1] = h_hist.GetBinCenter(i)
    Bern5_dist[i-1] = Bern5(mass[i-1])
    signal_dist[i-1] = signal(mass[i-1])
    bkg_plus_sig[i-1] = Bern5_dist[i-1] + signal_dist[i-1]
    h_bkg.SetBinContent(i,Bern5_dist[i-1])
    h_bkg.SetBinError(i,np.sqrt(Bern5_dist[i-1]))
    h_bkgsig.SetBinContent(i,Bern5_dist[i-1] + signal_dist[i-1])
    h_bkgsig.SetBinError(i,np.sqrt(Bern5_dist[i-1] + signal_dist[i-1]))


h_mean_best_Amplitude = np.zeros(len(lum))
h_mean_best_lengthscale = np.zeros(len(lum))

h_chi2_ge = np.zeros(Ntoys)
chi2_mean_ge = np.zeros(len(lum))
chi2_mean_ge_err = np.zeros(len(lum))
chi2_mean_par = np.zeros(len(lum))
chi2_mean_par_err = np.zeros(len(lum))

canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
canvas1.SetLeftMargin(0.125)
canvas1.SetBottomMargin(0.125)

"""Noise-free fit"""

for l in lum:


    print("=================================================================================")
    print("========================= Background fit to b-only ==============================")
    print("=================================================================================")
    toy = l*Bern5_dist
    h_bkg.Scale(l)

    Bkg_fit_function.SetParameters(1,1,-0.01,1e-6)
    Bkg_fit_function.FixParameter(0,1)
    fitresult = h_bkg.Fit(Bkg_fit_function,"SPRQ")
    canvas1.Update()

    if fitresult.Status() != 0:
        print("Fit failed!")

    h_chi2_par = fitresult.Chi2()/fitresult.Ndf()
    """
    THE LENGTH SCALE l MEANS THAT I NEED TO RESCALE THE X-AXIS BY THE LENGTH. 1->10 WITH l=1 MEANS THAT l=0.5 GIVES 1->5.
    WHAT DOES THAT MEAN FOR ME? l = 87 -> scale: x=1->2
    """
    lnprob = log_like_gp(mass,toy)
    minimumLLH, best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp(100,lnprob)
    kernel = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
    ge = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
    ge.compute(mass,yerr=np.sqrt(toy))
    covarmatrix = ge.get_matrix(mass)
    invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
    covmarprod = np.matmul(covarmatrix,invcovarmatrix)
    y_pred, y_var = ge.predict(toy,mass,return_var=True)

    h_mean_best_Amplitude[index] = best_fit_parameters[0]
    h_mean_best_lengthscale[index] = best_fit_parameters[1]

    Effndf[index] = covmarprod.trace()

    chi2 = np.sum((toy-y_pred)**2/y_pred)
    print('chi2',chi2,'effndf',covmarprod.trace(),'HPndf',len(ge.get_parameter_vector()))
    h_chi2_ge = chi2/(len(toy)-Effndf[index])
    print("Par",h_chi2_par,"GP",h_chi2_ge)
    h_bkg.Draw("")
    canvas1.Update()
    
    plt.figure(1)
    plt.plot(mass,toy-toy,color='r')
    plt.scatter(mass,toy-toy,c='k',marker='.')
    plt.plot(mass,toy-y_pred,'b-',label='Res GP')
    plt.fill_between(mass,toy-y_pred-np.sqrt(y_var),toy-y_pred+np.sqrt(y_var),color='k',alpha=0.5,label=r'$1\sigma$')
    plt.plot(mass,l*signal_dist,'c-.')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title(r'Background on b-only, Luminosity $\int Ldt = %.0f fb^{-1}$' % (36*l))
    plt.savefig(save_path + 'Residuals_bonly_lum%.0f.png'%(36*l))
    #plt.show()
    plt.close()
    
    plt.figure(2)
    plt.scatter(mass,toy,c='k',marker='.')
    plt.plot(mass,y_pred,color='b')
    plt.savefig(save_path + 'bonlydist%.0f.png'%(36*l))
    #plt.show()
    plt.close()

    plt.figure(3)
    plt.contourf(mass,mass,covarmatrix)
    plt.colorbar()
    plt.savefig(save_path + 'ContCov_lum%.0f'%(36*l))
    plt.close()

    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lum,h_mean_best_Amplitude,'b-')
    ax2.plot(lum,h_mean_best_lengthscale,'r-')
    ax1.set_xlabel("Luminosity scale factor")
    ax1.set_ylabel("Amplitude value")
    ax2.set_ylabel("Length scale value")
    plt.title("Amplitude and lengthscale evolution")
    plt.savefig(save_path + 'ParamEvoLum%.0f.png'%(36*l))
    #plt.show()
    plt.close()
    
    index += 1
    ajgjgjgjg = 251
    if ajgjgjgjg == 251:
        continue
    """
    
    
    print("=================================================================================")
    print("====================== Background + signal fit on b-only ========================")
    print("=================================================================================")


    Sig_fit_function.SetParameters(1,1,-0.01,1e-6,mean,sigma,Amp)
    Sig_fit_function.FixParameter(0,1)
    fitresult = h_bkg.Fit(Sig_fit_function,"SPRQ")
    canvas1.Update()
    if fitresult.Status() != 0:
        print("Fit failed!")

    h_chi2_par = fitresult.Chi2()/fitresult.Ndf()

    lnprob = log_like_gp_sig(mass,toy)
    minimumLLH, best_fit_parameters, best_fit_parameters_errors = fit_minuit_gp_sig(100,lnprob)
    kernel_bkg = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
    kernel_sig = best_fit_parameters[2]*george.kernels.LocalGaussianKernel(location=best_fit_parameters[3],log_width=best_fit_parameters[4])
    kernel = kernel_bkg+kernel_sig
    ge = george.GP(kernel=kernel,solver=george.HODLRSolver,mean=np.median(toy))
    ge.compute(mass,yerr=np.sqrt(toy))
    covarmatrix = ge.get_matrix(mass)
    invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
    covmarprod = np.matmul(covarmatrix,invcovarmatrix)
    y_pred, y_var = ge.predict(toy,mass,return_var=True)

    Effndf[index] = covmarprod.trace()

    chi2 = np.sum((toy-y_pred)**2/y_pred)
    print('chi2',chi2,'effndf',covmarprod.trace(),'HPndf',len(ge.get_parameter_vector()))
    h_chi2_ge = chi2/(len(toy)-Effndf[index])
    print("Par",h_chi2_par,"GP",h_chi2_ge)
    plt.figure(1)
    plt.plot(mass,toy-toy,color='r')
    plt.scatter(mass,toy-toy,c='k',marker='.')
    plt.plot(mass,toy-y_pred,'b-',label='Res GP')
    plt.fill_between(mass,toy-y_pred-np.sqrt(y_var),toy-y_pred+np.sqrt(y_var),color='k',alpha=0.5,label=r'$1\sigma$')
    plt.plot(mass,l*signal_dist,'c-.')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title(r'Signal$+$Background on b-only, Luminosity $\int Ldt = %.0f fb^{-1}$' % (36*l))
    plt.savefig(save_path + 'Residuals_sig_bonly_lum%.0f.png'%(36*l))
    #plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(mass,toy,c='k',marker='.')
    plt.plot(mass,y_pred,color='b')
    plt.savefig(save_path + 'sig_bonly_dist%.0f.png'%(36*l))
    #plt.show()
    plt.close()

    plt.figure(3)
    plt.contourf(mass,mass,covarmatrix)
    plt.colorbar()
    plt.savefig(save_path + 'ContCov_lum%.0f'%(36*l))
    plt.close()
    print("=================================================================================")
    print("======================= Background + signal fit on s+b ==========================")
    print("=================================================================================")


    toy = l*bkg_plus_sig
    h_bkgsig.Scale(l)
    Sig_fit_function.SetParameters(1,1,-0.01,1e-6,mean,sigma,Amp)
    Sig_fit_function.FixParameter(0,1)
    fitresult = h_bkgsig.Fit(Sig_fit_function,"SPRQ")
    canvas1.Update()

    if fitresult.Status() != 0:
        print("Fit failed!")
    h_chi2_par = fitresult.Chi2()/fitresult.Ndf()

    lnprob = log_like_gp_sig(mass,toy)
    minimumLLH, best_fit_parameters, best_fit_parameters_errors = fit_minuit_gp_sig(100,lnprob,fix=True)
    kernel_bkg = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
    kernel_sig = best_fit_parameters[2]*george.kernels.LocalGaussianKernel(location=best_fit_parameters[4],log_width=best_fit_parameters[3])
    kernel = kernel_bkg+kernel_sig
    ge = george.GP(kernel=kernel,solver=george.HODLRSolver,mean=np.median(toy))
    ge.compute(mass,yerr=np.sqrt(toy))
    covarmatrix = ge.get_matrix(mass)
    invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
    covmarprod = np.matmul(covarmatrix,invcovarmatrix)
    y_pred, y_var = ge.predict(toy,mass,return_var=True)

    print("SigAmp:",np.sqrt(best_fit_parameters[2]),"+/-",best_fit_parameters_errors[2]/(2*np.sqrt(best_fit_parameters[2])))
    Effndf[index] = covmarprod.trace()

    chi2 = np.sum((toy-y_pred)**2/y_pred)
    print('chi2',chi2,'effndf',covmarprod.trace(),'HPndf',len(ge.get_parameter_vector()))
    h_chi2_ge = chi2/(len(toy)-Effndf[index])
    print("Par",h_chi2_par,"GP",h_chi2_ge)    
    plt.figure(1)
    plt.plot(mass,toy-toy,color='r')
    plt.scatter(mass,toy-toy,c='k',marker='.')
    plt.plot(mass,toy-y_pred,'b-',label='Res GP')
    plt.fill_between(mass,toy-y_pred-np.sqrt(y_var),toy-y_pred+np.sqrt(y_var),color='k',alpha=0.5,label=r'$1\sigma$')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title(r'Signal$+$Background on s$+$b, Luminosity $\int Ldt = %.0f fb^{-1}$' % (36*l))
    plt.savefig(save_path + 'Residuals_sig_bkg_lum%.0f.png'%(36*l))
    #plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(mass,toy,c='k',marker='.')
    plt.plot(mass,y_pred,color='b')
    plt.savefig(save_path + 'sig_bkgdist%.0f.png'%(36*l))
    #plt.show()
    plt.close()

    
    
    index += 1



print("=================================================================================")
print("========================= Background fit to b-only ==============================")
print("=================================================================================")
toy = np.zeros(len(mass))

chi2_mean_gp = np.zeros(len(lum))
chi2_mean_gp_err = np.zeros(len(lum))
chi2_mean_par = np.zeros(len(lum))
chi2_mean_par_err = np.zeros(len(lum))
chi2 = np.zeros(Ntoys)
EffNdf = np.zeros(Ntoys)
EffNdf_mean = np.zeros(len(lum))
chi2_mean_noNDF = np.zeros(len(lum))
chi2_var_noNDF = np.zeros(len(lum))
chi2_fit = np.zeros(Ntoys)
h_chi2 = r.TH1D("h_chi2","Chi2 ad-hoc",100,0,20)
color = ['r','b','g','c','m','k','chartreuse']
index = 0
Error = 0

#canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
#canvas1.SetLeftMargin(0.125)
#canvas1.SetBottomMargin(0.125)

hs = r.THStack("hs","Chi2 ad-hoc")


for l in lum:
    for t in range(Ntoys):        
        for i_bin in range(1,h_hist.GetNbinsX()+1):
            toy[i_bin-1] = R.Poisson(l*Bern5_dist[i_bin-1])
            h_bkg.SetBinContent(i_bin,toy[i_bin-1])
            h_bkg.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))
        
        Bkg_fit_function.SetParameters(1,1,-0.01,1e-6)
        Bkg_fit_function.FixParameter(0,1)
        fitresult = h_bkg.Fit(Bkg_fit_function,"SPRQ")
        canvas1.Update()

        if fitresult.Status() != 0:
            print("Fit failed!")

        h_chi2_par = fitresult.Chi2()/fitresult.Ndf()
        h_chi2.Fill(h_chi2_par)

        lnprob = log_like_gp(mass,toy)
        minimumLLH, best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp(100,lnprob)
        kernel = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        ge = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        ge.compute(mass,yerr=np.sqrt(toy))
        covarmatrix = ge.get_matrix(mass)
        invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
        covmarprod = np.matmul(covarmatrix,invcovarmatrix)
        y_pred, y_var = ge.predict(toy,mass,return_var=True)

        EffNdf[t] = covmarprod.trace()

        chi2[t] = np.sum((toy-y_pred)**2/y_pred)
        chi2_fit[t] = chi2[t]/(len(toy) - EffNdf[t])

    chi2_mean_par[index] = h_chi2.GetMean()
    chi2_mean_par_err[index] = h_chi2.GetStdDev()
    EffNdf_mean[index] = np.mean(EffNdf)

    h_chi2.Reset()
    plt.hist(chi2,bins=50,color=color[index],label='Lum: %.0f'%l,alpha=0.8,histtype='step')
    chi2_mean_ge[index] = np.mean(chi2_fit)
    chi2_mean_ge_err[index] = np.std(chi2_fit)

    index += 1



plt.xlabel("Chi2")
plt.ylabel("#")
plt.legend()
plt.title("Test statistic distribution")
plt.savefig(save_path+'chi2_b_dist_lum%.0f.png'%(36*l))
plt.close()
plt.figure(2)
plt.plot(lum,np.linspace(1,1,len(lum)),'k')
plt.errorbar(lum,chi2_mean_ge,yerr=chi2_mean_ge_err,color='r',label='GP',marker='.')
plt.errorbar(lum,chi2_mean_par,yerr=chi2_mean_par_err,color='b',label='Ad-hoc',marker='.')
plt.xlabel('Luminosity scale factor')
plt.ylabel(r'$\chi^2/ndf$')
plt.title(r'Test statistic evolution')
plt.legend()
plt.savefig(save_path+'Lum%.0f_chi2_mean_b_b.png'%(36*l))
plt.close()

print("=================================================================================")
print("====================== Background + signal fit on b-only ========================")
print("=================================================================================")
index = 0

residuals = np.zeros((len(lum),len(mass)))

for l in lum:
    for t in range(Ntoys):        
        for i_bin in range(1,h_hist.GetNbinsX()+1):
            toy[i_bin-1] = R.Poisson(l*Bern5_dist[i_bin-1])
            h_bkg.SetBinContent(i_bin,toy[i_bin-1])
            h_bkg.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))

        Sig_fit_function.SetParameters(1,1,-0.01,1e-6,mean,sigma,Amp)
        Sig_fit_function.FixParameter(0,1)
        fitresult = h_bkg.Fit(Sig_fit_function,"SPRQ")
        canvas1.Update()

        if fitresult.Status() != 0:
            print("Fit failed!")

        h_chi2_par = fitresult.Chi2()/fitresult.Ndf()
        h_chi2.Fill(h_chi2_par)

        lnprob = log_like_gp_sig(mass,toy)
        minimumLLH, best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp_sig(100,lnprob)
        kernel_bkg = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        kernel_sig = best_fit_parameters[2]*george.kernels.LocalGaussianKernel(location=best_fit_parameters[4],log_width=best_fit_parameters[3])
        kernel = kernel_bkg + kernel_sig
        ge = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        ge.compute(mass,yerr=np.sqrt(toy))
        covarmatrix = ge.get_matrix(mass)
        invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
        covmarprod = np.matmul(covarmatrix,invcovarmatrix)
        y_pred, y_var = ge.predict(toy,mass,return_var=True)

        EffNdf[t] = covmarprod.trace()

        chi2[t] = np.sum((toy-y_pred)**2/y_pred)
        chi2_fit[t] = chi2[t]/(len(toy) - EffNdf[t])

        residuals[index][:] += toy - y_pred


    chi2_mean_par[index] = h_chi2.GetMean()
    chi2_mean_par_err[index] = h_chi2.GetStdDev()
    EffNdf_mean[index] = np.mean(EffNdf)

    h_chi2.Reset()
    #plt.hist(chi2,bins=50,color=color[index],label='Lum: %.0f'%l,alpha=0.8,histtype='step')
    chi2_mean_ge[index] = np.mean(chi2_fit)
    chi2_mean_ge_err[index] = np.std(chi2_fit)

    index += 1


"""
plt.xlabel("Chi2")
plt.ylabel("#")
plt.legend()
plt.title("Test statistic distribution")
plt.savefig(save_path+'chi2_b_dist_lum%.0f.png'%(36*l))
plt.close()
plt.figure(2)
plt.errorbar(lum,chi2_mean_ge,yerr=chi2_mean_ge_err,color='r',label='GP',marker='.')
plt.errorbar(lum,chi2_mean_par,yerr=chi2_mean_par_err,color='b',label='Ad-hoc',marker='.')
plt.xlabel('Luminosity scale factor')
plt.ylabel(r'$\chi^2/ndf$')
plt.legend()
plt.savefig(save_path+'Lum%.0f_chi2_mean_bps_b.png'%(36*l))
plt.close()
"""
for i in range(len(lum)):
    plt.figure(3)
    plt.plot(mass,np.zeros(len(mass)),color='r')
    plt.scatter(mass,np.zeros(len(mass)),c='k',marker='.')
    plt.plot(mass,residuals[i][:],'b-',label='Res GP')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title(r'Sig $+$ Bkg on b-only, Luminosity $\int Ldt = %.0f fb^{-1}$' % (36*lum[i]))
    plt.savefig(save_path + 'Summed_Residuals_sig_bonly_lum%.0f.png'%(36*lum[i]))
    #plt.show()
    plt.close()


print("=================================================================================")
print("======================= Background + signal fit on s+b ==========================")
print("=================================================================================")

index = 0
residuals = np.zeros((len(lum),len(mass)))
for l in lum:
    for t in range(Ntoys):        
        for i_bin in range(1,h_hist.GetNbinsX()+1):
            toy[i_bin-1] = R.Poisson(l*bkg_plus_sig[i_bin-1])
            h_bkgsig.SetBinContent(i_bin,toy[i_bin-1])
            h_bkgsig.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))

        Sig_fit_function.SetParameters(1,1,-0.01,1e-6,mean,sigma,Amp)
        Sig_fit_function.FixParameter(0,1)
        fitresult = h_bkgsig.Fit(Sig_fit_function,"SPRQ")
        canvas1.Update()

        if fitresult.Status() != 0:
            print("Fit failed!")

        h_chi2_par = fitresult.Chi2()/fitresult.Ndf()
        h_chi2.Fill(h_chi2_par)

        lnprob = log_like_gp_sig(mass,toy)
        minimumLLH, best_fit_parameters,best_fit_parameters_errors = fit_minuit_gp_sig(100,lnprob,fix=True)
        kernel_bkg = best_fit_parameters[0]*george.kernels.ExpSquaredKernel(metric=best_fit_parameters[1])
        kernel_sig = best_fit_parameters[2]*george.kernels.LocalGaussianKernel(location=best_fit_parameters[4],log_width=best_fit_parameters[3])
        kernel = kernel_bkg + kernel_sig
        ge = george.GP(kernel,solver=george.HODLRSolver,mean=np.median(toy))
        ge.compute(mass,yerr=np.sqrt(toy))
        covarmatrix = ge.get_matrix(mass)
        invcovarmatrix = np.linalg.inv(covarmatrix + toy*np.eye(len(mass)))
        covmarprod = np.matmul(covarmatrix,invcovarmatrix)
        y_pred, y_var = ge.predict(toy,mass,return_var=True)

        EffNdf[t] = covmarprod.trace()

        chi2[t] = np.sum((toy-y_pred)**2/y_pred)
        chi2_fit[t] = chi2[t]/(len(toy) - EffNdf[t])

        residuals[index] += toy - y_pred

    chi2_mean_par[index] = h_chi2.GetMean()
    chi2_mean_par_err[index] = h_chi2.GetStdDev()
    EffNdf_mean[index] = np.mean(EffNdf)

    h_chi2.Reset()
    #plt.hist(chi2,bins=50,color=color[index],label='Lum: %.0f'%l,alpha=0.8,histtype='step')
    chi2_mean_ge[index] = np.mean(chi2_fit)
    chi2_mean_ge_err[index] = np.std(chi2_fit)

    index += 1


"""
plt.xlabel("Chi2")
plt.ylabel("#")
plt.legend()
plt.title("Test statistic distribution")
plt.savefig(save_path+'chi2_b_dist_lum%.0f.png'%(36*l))
plt.close()
plt.figure(2)
plt.errorbar(lum,chi2_mean_ge,yerr=chi2_mean_ge_err,color='r',label='GP',marker='.')
plt.errorbar(lum,chi2_mean_par,yerr=chi2_mean_par_err,color='b',label='Ad-hoc',marker='.')
plt.xlabel('Luminosity scale factor')
plt.ylabel(r'$\chi^2/ndf$')
plt.legend()
plt.savefig(save_path+'Lum%.0f_chi2_mean_bps_bps.png'%(36*l))
plt.close()
"""
for i in range(len(lum)):
    plt.figure(3)
    plt.plot(mass,np.zeros(len(mass)),color='r')
    plt.scatter(mass,np.zeros(len(mass)),c='k',marker='.')
    plt.plot(mass,residuals[i][:],'b-',label='Res GP')
    plt.legend()
    plt.xlabel(r'$m_{\gamma\gamma}[GeV]$')
    plt.ylabel("Residuals")
    plt.title(r'Sig $+$ Bkg on s+b, Luminosity $\int Ldt = %.0f fb^{-1}$' % (36*lum[i]))
    plt.savefig(save_path + 'Summed_Residuals_sig_bkg_lum%.0f.png'%(36*lum[i]))
    #plt.show()
    plt.close()



"""
What I want my program to do:
    - fit the noise-free data set.
    - make and fit poisson distributed data set
    - with and without signal
    - and with and without signal-kernel
    - plots:
        - Chi2
        - residuals
        - the distribution with variance bands.
    
    - give parameters with uncertainty. Should be made easy to read and copy.
    - Choose a bunch of functions that can be fitted with noise, and plot the Chi2



    - calculate profile likelihood
    - Find p-values 

"""





