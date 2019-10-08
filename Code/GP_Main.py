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


R = r.TRandom(0)

if sys.argv[2] == "SignalOn" or sys.argv[2] == "signalon":
    SignalOn = True
else:
    SignalOn = False

if sys.argv[3] == "Sigmodel" or sys.argv[3] == "sigmodel":
    Sigmodel = True
else:
    Sigmodel = False

if sys.argv[4] == "epoly2":
    Epoly2_fit = True
    Bern5_fit = False
elif sys.argv[4] == "Bern5":
    Epoly2_fit = False
    Bern5_fit = True

def epoly2(vars,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(vars[0] - 100) + pars[3]*(vars[0]-100)*(vars[0]-100))
def epoly2_test(x,pars):
    return pars[0]*np.exp(pars[1] + pars[2]*(x - 100) + pars[3]*(x-100)*(x-100))


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


def neg_log_like(Amp,length):
    ge.set_parameter_vector((Amp,length))
    return -ge.log_likelihood(toy)

def grad_neg_log_like(p):
    ge.set_parameter_vector(p)
    return -ge.grad_log_likelihood(toy)


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

tf = r.TFile.Open("diphox_shape_withGJJJDY_WithEffCor.root")

tf.cd()
tf.ReadAll()

h_hist = tf.Get("Mgg_CP0").Clone()
binwidth = h_hist.GetBinWidth(1)
nbins = h_hist.GetNbinsX()
xmin = h_hist.GetXaxis().GetXmin()
xmax = h_hist.GetXaxis().GetXmax()
h_hist.Rebin(int(2./((xmax-xmin)/nbins)))

"""
Draw all histograms in every step
"""



""" Now we have read the file containing the histogram and rebinned for H->gamgam. To remove any poisson noise as much as possible
we fit the data with a Bernstein polynomial of degreee 5. This polynomial, with added Poisson noise, epoly2 should be able to fit
for small luminosities. Then we see if epoly2 and GP can follow for increasing Lum. We will also fit with other polynomials, like Bern 3 and 4."""

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
elif Epoly2_fit:
    fit_function = r.TF1("fit_function",epoly2,xmin,xmax,4)
    fit_function.SetParameters(1,1,-0.01,0)
    fit_function.SetParNames("Norm","a","b","c")
elif Bern5_fit:
    fit_function = r.TF1("fit_function",Bern,xmin,xmax,8)
    fit_function.SetParameters(1,0.1,0.01,0.001,0.0001,0.00001)
    fit_function.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
    fit_function.FixParameter(6,xmin)
    fit_function.FixParameter(7,xmax)

signal = r.TF1("signal",Gaussian,xmin,xmax,3)
signal.SetParameters(mean,sigma,Amp)
signal.SetParNames("Mean","Sigma","Amplitude")
h_toy = h_truth.Clone("h_toy")
h_toy.Reset()
lum = np.array([1,10,20,40,50,60,80,100])
lum = np.array([1,125,500,750,1000,2500,5000,7500,10000,12500])
h_chi2_ge = np.zeros(Ntoys)
h_chi2_param = np.zeros(Ntoys)
h_chi2_base = np.zeros(Ntoys)

chi2_lum_ge = np.zeros(len(lum))
chi2_lum_ge_err = np.zeros(len(lum))
chi2_lum_base = np.zeros(len(lum))
chi2_lum_base_err = np.zeros(len(lum))
chi2_lum_par = np.zeros(len(lum))
chi2_lum_par_err = np.zeros(len(lum))

h_mean_best_Amplitude = np.zeros(len(lum))
h_mean_best_lengthscale = np.zeros(len(lum))
h_best_Amplitude = np.zeros(Ntoys)
h_best_lengthscale = np.zeros(Ntoys)

mass = np.zeros(h_truth.GetNbinsX())
toy = np.zeros(h_truth.GetNbinsX())
truth = np.zeros(h_truth.GetNbinsX())
Error = 0
index = 0
Overfit = 0
res = np.zeros(h_truth.GetNbinsX())

canvas1 = r.TCanvas("canvas1","Standard Canvas",600,400)
canvas1.SetLeftMargin(0.125)
canvas1.SetBottomMargin(0.125)

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
            h_toy.SetBinError(i_bin,np.sqrt(toy[i_bin-1]))
        
        if Epoly2_fit:
            fit_function.SetParameters(1,1,-0.01,1e-6)
            fit_function.FixParameter(0,1)
        elif Bern5_fit:
            #fit_function.SetParameters(100,100,100,100,100,100)
            fit_function.SetParameters(1000,1000,100,100,100,200)
            fit_function.SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax")
            fit_function.FixParameter(6,xmin)
            fit_function.FixParameter(7,xmax)
        

        fitresults = h_toy.Fit(fit_function,"SPR")
        
        if fitresults.Status() != 0:
            Error += 1


        h_chi2_param[t] = fitresults.Chi2()/fitresults.Ndf()


        """George"""
        lnprob = log_like_gp(mass,toy)
        minimumLLH, best_fit_params = fit_minuit_gp(100,lnprob)
        kernel_ge = best_fit_params[0]*george.kernels.ExpSquaredKernel(metric=best_fit_params[1])#,block=(1,10))
        ge = george.GP(kernel_ge,solver=george.HODLRSolver,mean=np.median(toy))#,white_noise=np.log(np.sqrt(np.mean(toy))))
        ge.compute(mass,yerr=np.sqrt(toy))
        print(ge.get_parameter_vector())

        h_best_Amplitude[t] = best_fit_params[0]
        h_best_lengthscale[t] = best_fit_params[1]
        #m = minimize(neg_log_like,ge.get_parameter_vector(),jac=grad_neg_log_like)#,bounds=((1,1000),(6,15)))
        
        #plog.logl_landscape(toy,mass,ge)        
        print(ge.get_parameter_vector())
        y_pred, y_var = ge.predict(toy,mass,return_var = True)
        chi2_ge = np.sum((toy-y_pred)**2/y_pred)
        h_chi2_ge[t] = chi2_ge/(len(toy) - len(ge.get_parameter_vector()))
        print("George Chi2/ndf",h_chi2_ge[t],"Ad-hoc Chi2/ndf",h_chi2_param[t])


        res = y_pred - toy







        
         
        if h_chi2_ge[t] < 0.01:
            Overfit += 1 

        h_toy.Draw("pe")
        canvas1.Update()



        plt.clf()
        plt.scatter(mass,toy,c='r',alpha=0.8)
        plt.plot(mass,y_pred,'b-')
        plt.fill_between(mass,y_pred-np.sqrt(y_var),y_pred+np.sqrt(y_var),color='g',alpha=0.2)
        #plt.pause(0.05)
        
    h_mean_best_Amplitude[index] = np.mean(h_best_Amplitude)
    h_mean_best_lengthscale[index] = np.mean(h_best_lengthscale)
    chi2_lum_ge[index] = np.mean(h_chi2_ge)
    chi2_lum_par[index] = np.mean(h_chi2_param)

    index += 1


def expo(x,a,b):
    return a*np.exp(b*x)

def poly(x,a,b,c):
    return a*x**2 + b*x + c

def poly3(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

def lin(x,a,b):
    return a*x + b

h_mean_best_Amplitude_log = np.log(h_mean_best_Amplitude)
h_mean_best_lengthscale_log = np.log(h_mean_best_lengthscale)
popt_exp, pcov_exp = curve_fit(expo,lum,h_mean_best_Amplitude,[np.exp(10),0.01])
popt_poly, pcov_poly = curve_fit(poly,lum,h_mean_best_Amplitude,[np.exp(10),np.exp(5),np.exp(10)])
popt_poly3, pcov_poly3 = curve_fit(poly3,lum,h_mean_best_Amplitude,[np.exp(10),np.exp(5),np.exp(10),np.exp(5)])
#popt_lin, pcov_lin = curve_fit(lin,lum,h_mean_best_Amplitude,[])
#print(popt_poly)

#print(np.mean(chi2_lum_ge))
plt.figure(2)
plt.plot(lum,chi2_lum_ge,marker=".",label='GP',c='b')
plt.plot(lum,chi2_lum_par,marker=".",label='Ad hoc',c='r')

plt.xlabel("Lum scale")
plt.ylabel(r'$\chi^2$/ndf')
plt.legend()
plt.title("Chi2/ndf Ad-hoc and GP")
plt.figure(3)
plt.plot(lum,chi2_lum_ge,marker=".",label='GP',c='r')

plt.xlabel("Lum scale")
plt.ylabel(r'$\chi^2$/ndf')
plt.legend()
plt.title("Chi2/ndf Ad-hoc")

plt.figure(4)
plt.plot(lum,h_mean_best_Amplitude,marker='o',c='b')
plt.plot(lum,expo(lum,*popt_exp),'r-',label='exp')
plt.plot(lum,poly(lum,*popt_poly),'g--',label='poly')
plt.plot(lum,poly3(lum,*popt_poly3),'c-.',label='Poly3')
plt.xlabel("Luminosity scale factor")
plt.ylabel("Amp")
plt.legend()

plt.figure(5)
#plt.plot(lum,h_mean_best_lengthscale,marker='o')
#plt.xlabel("Luminosity scale factor")
#plt.ylabel("Lengthscale")
plt.plot(lum,h_mean_best_Amplitude_log,marker='o',c='b')
plt.xlabel("Luminosity scale factor")
plt.ylabel("Amp")
#plt.legend()


plt.show()

print("Overfitted: ", Overfit)
