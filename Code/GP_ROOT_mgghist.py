import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from  scipy.interpolate import interp1d
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import normalize

R = r.TRandom(0)

SignalOn = True
SignalOn = False

sigmodel = True
sigmodel = False

def background_function(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0]-100) + pars[3]*(vars[0]-100)*(vars[0]-100))
    
def signal_hist(luminosity=1):
    mini = np.array([110,115,120,125,130,135,140,145,150])
    cpin = [80.644,82.96,82.54,79.583,74.231,67.073,57.996,47.364,35.471]
    sighist = r.TH1D("signal","Signal", 50,100,199)
    signal = interp1d(mini,cpin,kind='cubic')
    for ibin in range(1,sighist.GetNbinsX()+1):
        binmass = sighist.GetBinCenter(ibin)
        if binmass >= 110 and binmass<=150:
            sighist.SetBinContent(ibin,signal(sighist.GetBinCenter(ibin))*luminosity)
    return sighist


def signal_function(vars, pars):
    return pars[2]/(np.sqrt(2*np.pi)*pars[1])*np.exp(-0.5*((vars[0]-pars[0])/pars[1])**2)


tf = r.TFile.Open("diphox_shape_withGJJJDY_WithEffCor.root")

tf.cd()
tf.ReadAll()

h_hist = tf.Get("Mgg_CP0").Clone()
binwidth = h_hist.GetBinWidth(1)
nbins = h_hist.GetNbinsX()
xmin = h_hist.GetXaxis().GetXmin()
xmax = h_hist.GetXaxis().GetXmax()
minmass = r.Double(100)
maxmass = r.Double(160)

h_hist.Rebin(int(2./((xmax-xmin)/nbins)))
nbins = h_hist.GetNbinsX()
binwidth = h_hist.GetBinWidth(1)

Ntoys = 1000

background = r.TF1("background",background_function,xmin,xmax,4)
background.SetParameters(1,2,-0.03,0)
background.SetParNames("Norm","a","b","c")
#signal = r.TF1("signal",signal_function,xmin,xmax,3)
#signal.SetParameters(mean,sigma,Amp)
#signal.SetParNames("Mean","Sigma","Amplitude")
h_signal = signal_hist()

hdiff = r.TH1D("hdiff","Residuals",nbins,xmin,xmax)
"""
hdiff2 = hdiff.Clone("hdiff2")
hdiff1sigmaNoFunc = hdiff.Clone("hdiff1sigmaNoFunc")
hdiff2sigmaNoFunc = hdiff.Clone("hdiff2sigmaNoFunc")


hdiff1sigma = h_hist.Clone("hdiff1sigma")
hdiff2sigma = h_hist.Clone("hdiff2sigma")
r.TVirtualFitter.GetFitter().GetConfidenceIntervals(hdiff1sigma,0.6827)
r.TVirtualFitter.GetFitter().GetConfidenceIntervals(hdiff2sigma,0.9545)
"""
backgroundfitresults = h_hist.Fit(background,"S","",minmass,maxmass)
diffc = r.TCanvas("diffc","Difference canvas")
diffc.cd()

for i in range(1, nbins+1):
    dy = h_hist.GetBinContent(i) - background.Eval(h_hist.GetBinCenter(i))
    #dy = background.Eval(h_hist.GetBinCenter(i))
    hdiff.SetBinContent(i,dy)

#h_hist.Draw("pe")
hdiff.Draw("pe")


input("Press enter to exit!")
































h_1 = r.TH1F("h_1","Standard histogram",50,100,199)
h_1.SetMarkerStyle(8)


kernel = 1.0 * RBF(length_scale=1.0,length_scale_bounds=(0.5,2))
mass = np.zeros(h_1.GetNbinsX())
toy = np.zeros(h_1.GetNbinsX())
h_chi2 = r.TH1D("h_chi2","",199,0,1)
h_loglik = r.TH1D("h_loglik","",100,-24000,-22000)
stat = 0
stat_diff_mean = 0

for t in range(Ntoys):
    for i_bin in range(h_1.GetNbinsX()):
        m = h_1.GetBinCenter(i_bin)
        mass[i_bin] = m
        mu_bin = background(m)
        if SignalOn:
            mu_bin += signal(m)
        toy[i_bin] = R.Poisson(mu_bin)
        h_1.SetBinContent(i_bin,toy[i_bin])


    gp = GaussianProcessRegressor(kernel=kernel, alpha=np.sqrt(toy))
    gp.fit(mass.reshape(-1,1),toy)
    y_mean = gp.predict(mass.reshape(-1,1))
    h_loglik.Fill(gp.log_marginal_likelihood())
    h_chi2.Fill(np.sum((toy-y_mean)**2/y_mean))
    stat += (y_mean-toy)/np.sqrt(toy)
    stat_diff_mean += stat*stat

std_ave = np.sqrt(stat_diff_mean/(Ntoys - 1))
stat_sum = stat/float(Ntoys)
std_sum = std_ave/float(Ntoys)

plt.scatter(mass,stat_sum,c='r',alpha=0.8,s=20)
plt.fill_between(mass,stat_sum-std_sum,stat_sum+std_sum,alpha=0.2,color='k')
#plt.show()

#h_chi2.Draw()
h_loglik.Draw()
#h_1.Draw("PE")
input("Press enter to exit!")







