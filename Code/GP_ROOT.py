import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import normalize

R = r.TRandom(0)

SignalOn = True
#SignalOn = False

sigmodel = True
#sigmodel = False

def background_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))

def background_fit_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))


def signal_function(vars, pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)

A = 1
a = 10.3308
b = -0.0241945
c = 7.58985e-6
mean = 150
sigma = 1.
Amp = 1
lum = [1]# np.array([0.5,1,2,5,10,15,20,25,40,50,60,80,100])


xmin, xmax = r.Double(100), r.Double(199)
Ntoys = 1

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



kernel = 1.0*RBF(length_scale=1.0,length_scale_bounds=(0.5,1.5))
mass = np.zeros(h_1.GetNbinsX())
toy = np.zeros(h_1.GetNbinsX())
h_chi2_gp = r.TH1D("h_chi2_gp","",200,0,10000)
h_chi2_param = h_chi2_gp.Clone("h_chi2_param")

chi2_lum_gp = np.zeros(len(lum))
chi2_lum_gp_err = np.zeros(len(lum))
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


for l in lum:
    for t in range(Ntoys):
        for i_bin in range(h_1.GetNbinsX()):
            m = h_1.GetBinCenter(i_bin)
            mass[i_bin] = m
            mu_bin = background(m)
            if SignalOn:
                mu_bin += signal(m)
            toy[i_bin] = R.Poisson(l*mu_bin)
            h_1.SetBinContent(i_bin,toy[i_bin])
        #print(toy)
        background_fit.SetParameter(1,2)
        background_fit.SetParameter(2,-0.1)
        background_fit.SetParameter(3,1e-9)
        fitresults = h_1.Fit(background_fit,"SNQ")
        if fitresults.Status() != 0:
            print("Error in fit")
            Error += 1
        print(t+1)
        h_chi2_param.Fill(fitresults.Chi2()/fitresults.Ndf())
        #print(fitresults.Chi2()/fitresults.Ndf())
        gp = GaussianProcessRegressor(kernel=kernel, alpha=np.sqrt(toy))
        gp.fit(mass.reshape(-1,1),toy)

        y_mean = gp.predict(mass.reshape(-1,1))
        chi2_gp = np.sum((toy-y_mean)**2/toy)
        h_chi2_gp.Fill(chi2_gp/(len(toy) - 1 - gp.kernel_.n_dims))
        #print(chi2_gp/(len(toy) - 1 - len(gp.kernel_.theta)))
        #h_loglik.Fill(gp.log_marginal_likelihood())
        
        #h_1.Draw("PE")
        #canvas1.Update()
        #input("Press enter to exit!")
        foobar





    chi2_lum_gp[index] = h_chi2_gp.GetMean()
    chi2_lum_par[index] = h_chi2_param.GetMean()
    chi2_lum_gp_err[index] = h_chi2_gp.GetMeanError()
    chi2_lum_par_err[index] = h_chi2_param.GetMeanError()
    index += 1
    #plt.plot(mass,y_mean,'k',lw=2,zorder=9)
    #plt.scatter(mass,toy,c='b', alpha=0.5,s=20,zorder=11,edgecolors=(0,0,0))
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


plt.errorbar(lum,chi2_lum_gp,yerr=chi2_lum_gp_err,marker="o",label='GP')
plt.errorbar(lum,chi2_lum_par,yerr=chi2_lum_par_err,marker="o",label='ad-hoc')
plt.xlabel("Luminosity scale factor")
plt.ylabel(r'$\chi^2$/ndf')
plt.yticks(np.arange(-1,max(chi2_lum_gp.max(),chi2_lum_par.max()))+1)
plt.legend()
plt.show()

