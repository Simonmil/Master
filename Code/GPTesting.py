import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import signal
import pylab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel
from sklearn.preprocessing import normalize




#np.random.seed(10)

def background(x, p0, p1, p2, A):
	return A*np.exp(p0 + p1*(x-100) + p2*(x-100)*(x-100))

p0 = 10.3308
p1 = -0.0241945
p2 = 7.58985e-6

mass = np.linspace(100,200,99)[:,np.newaxis]
mass_pred = np.random.uniform(100,200,10)[:,np.newaxis]
A = 1.0
events = background(mass,p0,p1,p2,A)# + 1000*np.random.normal(150,4)
events_pred	 = background(mass_pred,p0,p1,p2,A)
#sig = 1000*signal.gaussian(100,1)[:,np.newaxis]
#events_sig = events + sig

toy = np.zeros(len(mass))

noise = 0.01
scalefactor= 1.0
Ntoys = 1000

kernel = 1.0 * RBF(length_scale=1.0,length_scale_bounds=(1e-5,10)) 
kernel = Kernel()

gp = GaussianProcessRegressor(kernel=kernel,alpha=np.sqrt(events), normalize_y=True)
plt.figure(1,figsize=(8,8))

y_mean, y_std = gp.predict(mass, return_std=True)
plt.plot(mass,y_mean,'k',lw=2,zorder=9)
plt.fill_between(mass[:,0],y_mean-y_std,y_mean+y_std, alpha=0.2,color='k')
y_samples = gp.sample_y(mass,3)
plt.plot(mass,y_samples,lw=1)

plt.title("Prior (kernel:    %s)" % kernel, fontsize=10)
plt.show()

r2score = np.zeros((Ntoys,len(mass)))
sum_of_toys = np.zeros(len(mass))
stat = np.zeros(len(mass))
#std_sqrt_sum = np.zeros(len(mass))
sqrd_diff_mean = np.zeros(len(mass))



for j in range(Ntoys):
	for i in range(len(mass)):
		mu_bin = events[i]
		Nevt_bin = np.random.poisson(mu_bin)*scalefactor
		toy[i] = Nevt_bin #*np.random.normal(1,noise)	
	gp = GaussianProcessRegressor(kernel=kernel,alpha=np.sqrt(toy),normalize_y=True)
	gp.fit(mass,toy)
	y_mean, y_std = gp.predict(mass, return_std=True)
	stat += (y_mean - events[:,0])/np.sqrt(events[:,0])     #y_std**2
	sqrd_diff_mean += stat**2

	#std_sqrt_sum += y_std**2
	r2score[j] = gp.score(mass,events)
	print(j)


std_ave = np.sqrt(sqrd_diff_mean/(Ntoys-1))
r2score = round(np.mean(r2score), 6)


plt.figure(2,figsize=(8,4))
plt.plot(mass,y_mean,'k',lw=2,zorder=9)
plt.fill_between(mass[:,0],y_mean-y_std, y_mean+y_std, alpha=0.2,color='k')
plt.scatter(mass[:,0],toy,c='b', alpha=0.5,s=20,zorder=11,edgecolors=(0,0,0), label= ("R2 score: " + str(r2score)))
plt.title("Posterior (kernel:  %s)" % gp.kernel_)
plt.legend()

y_samples = gp.sample_y(mass,5)
plt.plot(mass,y_samples,lw=1)


stat_sum = stat/float(Ntoys)
std_sum = std_ave/float(Ntoys)




plt.figure(3,figsize=(8,4))

#plt.subplot(2,1,1)
plt.plot(mass, np.zeros(len(mass)),'k',lw=1)
#plt.ylim(-1.5,1.5)
#plt.errorbar(mass[:,0],stat_sum,yerr=std_ave)
plt.scatter(mass[:,0],stat_sum,c='r', alpha=0.8,s=20)
plt.fill_between(mass[:,0],stat_sum - std_sum, stat_sum + std_sum, alpha=0.2, color='k')
plt.title("Residuals in the fit, %i toys." % Ntoys)
plt.ylabel("Residuals")

#plt.subplot(2,1,2)

plt.tight_layout()
plt.show()


"""
Neste steg:

Lage et plot, skal helst vare flat. Legg til usikkerheter.

Videre: 

Se pa standardavvik, mot bias. 

Legge inn signal i bakgrunnfunksjonen, og inn i kernel. Da ser vi på hvordan GP tilpasser. 

Hva betyr en lengdeskala på 0.1 når bin er 1? 

trekke ut y-sample etter tilpassning. 

Lette inn støy i kernel. Bare å plusse på?

Hva betyr funksjonen og GP osv?! OMGOMGOGM


Til presentasjonen:

Litt intro om bakenforliggende teori. Så litt om hva vi har gjort, deretter vise resultater.

22.03.1:

Se på usikkerheter i residuals


"""

