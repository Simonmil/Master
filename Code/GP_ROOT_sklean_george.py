import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.optimize import minimize
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import normalize
import george

R = r.TRandom(0)

SignalOn = True
SignalOn = False

sigmodel = True
#sigmodel = False

def background_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))

def background_fit_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))


def signal_function(vars, pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)

def neg_log_like(p):
    ge.set_parameter_vector(p)
    return -ge.log_likelihood(toy)

def grad_neg_log_like(p):
    ge.set_parameter_vector(p)
    return -ge.grad_log_likelihood(toy)

A = 1
a = 10.3308
b = -0.0241945
c = 7.58985e-6
mean = 150
sigma = 1.
Amp = 1
lum = np.array([0.5,1,2,5,10,15,20,25,40,50,60,80,100])


xmin, xmax = r.Double(100), r.Double(199)
Ntoys = 10

background = r.TF1("background",background_function,100,199,4)
background_fit = r.TF1("background",background_fit_function,100,199,4)
background.SetParameters(A,a,b,c)
background_fit.SetParameters(A,a,b,c)
background.FixParameter(0,1)
background_fit.FixParameter(0,1)
background.SetParNames("Norm","a","b","c")
background_fit.SetParNames("Norm","a","b","c")
signal = r.TF1("signal",signal_function,100,199,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")

h_1 = r.TH1F("h_1","Standard histogram",50,100,199)
h_1.SetMarkerStyle(8)


kernel_ge = 1.0*george.kernels.ExpSquaredKernel(metric=1.0)
kernel_sk = 1.0*RBF(length_scale=1.0,length_scale_bounds=(0.5,1.5))
mass = np.zeros(h_1.GetNbinsX())
toy = np.zeros(h_1.GetNbinsX())
truth = np.zeros(h_1.GetNbinsX())
h_chi2_sk = r.TH1D("h_chi2_sk","",200,0,10000)
h_chi2_ge = h_chi2_sk.Clone("h_chi2_ge")
h_chi2_param = h_chi2_sk.Clone("h_chi2_param")

chi2_lum_sk = np.zeros(len(lum))
chi2_lum_sk_err = np.zeros(len(lum))
chi2_lum_ge = np.zeros(len(lum))
chi2_lum_ge_err = np.zeros(len(lum))
chi2_lum_par = np.zeros(len(lum))
chi2_lum_par_err = np.zeros(len(lum))

h_loglik = r.TH1D("h_loglik","",100,-24000,-22000)
stat = 0
stat_diff_mean = 0
Error = 1
index = 0

#canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
#canvas1.SetLeftMargin(0.125)
#canvas1.SetBottomMargin(0.125)

lengthscale = np.zeros(len(lum))
metric_george = np.zeros(len(lum))
log_lik_sk = np.zeros(Ntoys)
log_lik_ge = np.zeros(Ntoys)
log_mean_ge = np.zeros(len(lum))
log_mean_sk = np.zeros(len(lum))

for l in lum:
    for t in range(Ntoys):
        for i_bin in range(h_1.GetNbinsX()):
            m = h_1.GetBinCenter(i_bin)
            mass[i_bin] = m
            mu_bin = background(m)
            if SignalOn:
                mu_bin += signal(m)
            truth[i_bin] = l*mu_bin
            toy[i_bin] = R.Poisson(l*mu_bin)
            h_1.SetBinContent(i_bin,toy[i_bin])
        #print(toy)
        background_fit.SetParameter(1,2)
        background_fit.SetParameter(2,-0.1)
        background_fit.SetParameter(3,1e-9)
        fitresults = h_1.Fit(background_fit,"SNQ")
        if fitresults.Status() != 0:
            Error += 1
        print(t+1)
        h_chi2_param.Fill(fitresults.Chi2()/fitresults.Ndf())
        #print(fitresults.Chi2()/fitresults.Ndf())
        kernel_ge = np.median(toy)*george.kernels.ExpSquaredKernel(metric=1.0)#, block=(0.5,1.5))
        ge = george.GP(kernel_ge,solver=george.HODLRSolver)#,mean=background,fit_mean=True)
        ge.compute(mass,yerr=np.sqrt(toy))
        m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)
        ge.set_parameter_vector(m.x)
        print(m)
        y_pred, y_cov = ge.predict(toy,mass)
        chi2_ge = np.sum((toy-y_pred)**2/toy)
        h_chi2_ge.Fill(chi2_ge/(len(toy) - 1 - len(ge.get_parameter_vector())))

        kernel_sk = np.median(toy)*RBF(length_scale=1.0,length_scale_bounds=(0.5,1.5))
        sk = GaussianProcessRegressor(kernel=kernel_sk, alpha=np.sqrt(toy))
        #sk.set_params(kernel__k1__constant_value = m.x[0],kernel__k2__length_scale = m.x[1])


        sk.fit(mass.reshape(-1,1),toy)

        y_mean = sk.predict(mass.reshape(-1,1))
        chi2_sk = np.sum((toy-y_mean)**2/toy)
        h_chi2_sk.Fill(chi2_sk/(len(toy) - 1 - sk.kernel_.n_dims))

        log_lik_ge[t] = ge.log_likelihood(toy)
        log_lik_sk[t] = sk.log_marginal_likelihood()

        #print(chi2_gp/(len(toy) - 1 - len(gp.kernel_.theta)))
        #h_loglik.Fill(gp.log_marginal_likelihood())
        
        #h_1.Draw("PE")
        #canvas1.Update()
        #input("Press enter to exit!")
    
    metric_george[index] = ge.get_parameter_vector()[1]
    lengthscale[index] = sk.kernel_.theta[1]
    log_mean_ge[index] = np.mean(log_lik_ge)
    log_mean_sk[index] = np.mean(log_lik_sk)


    chi2_lum_sk[index] = h_chi2_sk.GetMean()
    chi2_lum_ge[index] = h_chi2_ge.GetMean()
    chi2_lum_par[index] = h_chi2_param.GetMean()
    chi2_lum_sk_err[index] = h_chi2_sk.GetMeanError()
    chi2_lum_ge_err[index] = h_chi2_ge.GetMeanError()
    chi2_lum_par_err[index] = h_chi2_param.GetMeanError()
    index += 1
    plt.figure(index)
    plt.plot(mass,truth/truth,'b',lw=2)
    plt.plot(mass,y_pred/truth,'k',lw=2,label='GP george')
    plt.plot(mass,y_mean/truth,'r',lw=1.5,label='GP Sklearn')
    plt.scatter(mass,toy/truth,c='b', alpha=0.5,s=20,zorder=11,edgecolors=(0,0,0))
    plt.title("Lum-scale: %.3f" % l)
    plt.legend()

plt.show()

print(log_mean_ge)
print(log_mean_sk)

#plt.plot(lum,log_mean_ge,label='George')
#plt.plot(lum,log_mean_sk,label='SK')
#plt.legend()
#plt.show()

#plt.plot(lum,metric_george,label='length scale George')
#plt.plot(lum,lengthscale,label='length scale Sklearn')
#plt.legend()
#plt.show()

#foobar

print("Number of fits errors: ",Error)



#plt.scatter(mass,stat_sum,c='r',alpha=0.8,s=20)
#plt.fill_between(mass,stat_sum-std_sum,stat_sum+std_sum,alpha=0.2,color='k')
#plt.show()

#h_chi2_gp.GetXaxis().SetRange(0,h_chi2_gp.GetBinCenter)
#h_chi2_gp.Draw()
#canvas1.Update()
#input("Press enter to exit!")

#h_chi2_param.SetLineColor(2)
#h_chi2_param.Draw()
#canvas1.Update()
#h_loglik.Draw()
#input("Press enter to exit!")
#h_1.Draw("PE")
#canvas1.Update()
#input("Press enter to exit!")


plt.errorbar(lum,chi2_lum_sk,yerr=chi2_lum_sk_err,marker="o",label='GP Sklearn')
plt.errorbar(lum,chi2_lum_ge,yerr=chi2_lum_ge_err,marker="o",label='GP George')
plt.errorbar(lum,chi2_lum_par,yerr=chi2_lum_par_err,marker="o",label='ad-hoc')
plt.xlabel("Luminosity scale factor")
plt.ylabel(r'$\chi^2$/ndf')
#plt.yticks(np.arange(-1,max(chi2_lum_sk.max(),chi2_lum_par.max()))+1)
plt.legend()
plt.show()

"""




Check the code for the article and see how they initialize GP. And minuit?
"""