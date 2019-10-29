import numpy as np
import matplotlib.pyplot as plt
import george


x = np.random.uniform(-10,10,25)
x_ = np.linspace(-10,10,1000)


kernel = george.kernels.ExpSquaredKernel(metric=1)
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(x,yerr=0.1)
y = gp.sample(x)
y_pred_1,y_var_1 = gp.predict(y,x_,return_var=True)

kernel = george.kernels.ExpSquaredKernel(metric=0.1)
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(x,yerr=0.1)
y_pred_01,y_var_01 = gp.predict(y,x_,return_var=True)

kernel = george.kernels.ExpSquaredKernel(metric=3)
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(x,yerr=0.1)
y_pred_3,y_var_3 = gp.predict(y,x_,return_var=True)

kernel = george.kernels.Matern32Kernel(metric=1)
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(x,yerr=0.1)
y_pred_M32, y_var_M32 = gp.predict(y,x_,return_var=True)

kernel = george.kernels.RationalQuadraticKernel(log_alpha=1,metric=1)
gp = george.GP(kernel=kernel,solver=george.HODLRSolver)
gp.compute(x,yerr=0.1)
y_pred_RQ, y_var_RQ = gp.predict(y,x_,return_var=True)


plt.figure(1)
plt.scatter(x,y,marker='.',color='b')
plt.plot(x_,y_pred_1,'k-')
plt.fill_between(x_,y_pred_1-y_var_1,y_pred_1+y_var_1,alpha=0.5,color='g')
plt.title('ExpSquaredKernel: Metric=1')


plt.figure(2)
plt.scatter(x,y,marker='.',color='b')
plt.plot(x_,y_pred_01,'k-')
plt.fill_between(x_,y_pred_01-y_var_01,y_pred_01+y_var_01,alpha=0.5,color='g')
plt.title('ExpSquaredKernel: Metric=0.1')

plt.figure(3)
plt.scatter(x,y,marker='.',color='b')
plt.plot(x_,y_pred_3,'k-')
plt.fill_between(x_,y_pred_3-y_var_3,y_pred_3+y_var_3,alpha=0.5,color='g')
plt.title('ExpSquaredKernel: Metric=3')

plt.figure(4)
plt.scatter(x,y,marker='.',color='b')
plt.plot(x_,y_pred_M32,'k-')
plt.fill_between(x_,y_pred_M32-y_var_M32,y_pred_M32+y_var_M32,alpha=0.5,color='g')
plt.title('Matern 3/2: metric=1')

plt.figure(5)
plt.scatter(x,y,marker='.',color='b')
plt.plot(x_,y_pred_RQ,'k-')
plt.fill_between(x_,y_pred_RQ-y_var_RQ,y_pred_RQ+y_var_RQ,alpha=0.5,color='g')
plt.title('Rational Quadratic: metric=1, log_alpha=1')


plt.show()


