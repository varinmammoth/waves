#%%
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
# %%
class squarewave:
    def __init__(self, tau, tmax, A):
        self.tau = tau
        self.A = A
        self.tmax = tmax

        self.t = np.linspace(0,self.tmax,1000)
        self.wave = (A/2)*sp.square(2 * np.pi * t / tau) + A/2

    def get_coeff(self, n_coeff):
        coeff_list = []
        for n in range(1,n_coeff+1):
            coeff_list.append((-self.A)*((-1)**n - 1)/(n*np.pi))
        return coeff_list

    def plot(self, n_coeff, plotActual=True, save=False):
        coeff_list = self.get_coeff(n_coeff)
        sum = 0
        for n in range(1, n_coeff):
            sum += coeff_list[n-1]*np.sin((2*np.pi*n*self.t)/(self.tau))
        sum += self.A/2
        
        plt.plot(self.t, sum, label=n_coeff-1)
        if plotActual:
            plt.plot(self.t, self.wave, c='black')
        
        if save == True:
            return sum
# %%
sq1 = squarewave(240, 1000, 1)

n_coeff = [2, 10, 100]
for i in n_coeff:
    sq1.plot(i)
plt.legend(loc=1, title='# of terms')
plt.show()

# %%
def trap_int(x, y):
    sum = 0
    for i in range(0,len(y)-1):
        sum += y[i]*(x[i+1]-x[i]) - 0.5*(y[i]-y[i+1])*(x[i+1]-x[i])
    return sum
# %%
x1, y1 = np.loadtxt("Task1.3_Semicircle_high.txt", unpack=True, skiprows=1)
trap_integral = trap_int(x1, y1)
actual_integral = np.pi/2
print('Difference (high res): ', actual_integral-trap_integral)

x2, y2 = np.loadtxt("Task1.3_Semicircle_low.txt", unpack=True, skiprows=1)
trap_integral = trap_int(x2, y2)
actual_integral = np.pi/2
print('Difference (low res): ', actual_integral-trap_integral)
# %%
x1, y1 = np.loadtxt("Task1.3_sine_high.txt", unpack=True, skiprows=1)
trap_integral = trap_int(x1, y1)
actual_integral = 0
print('Difference (high res): ', actual_integral-trap_integral)

x2, y2 = np.loadtxt("Task1.3_sine_low.txt", unpack=True, skiprows=1)
trap_integral = trap_int(x2, y2)
actual_integral = 0
print('Difference (low res): ', actual_integral-trap_integral)
# %%
