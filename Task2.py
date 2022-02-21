#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import Task1 as t1
#%%
x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True, skiprows=1)
x1 = x1*1e-1
#%%
plt.plot(x1,y1,'.',label='4 min Data')
sq1 = t1.squarewave(4*60,x1[-1],100)
sq1.plot_actual(label='Initial square wave')
sq1.plot(2)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.show()
# %%
def get_D(w, y, r):
    denom = (np.log(y)/r)**2
    denom = 2*denom
    D = w/denom
    return D

#print(get_D((2*np.pi)/240, 0.1817, 7.88e-3))

def get_D_phaselag(w, r, phi):
    numer = w*(r**2)
    denom = 2*(phi**2)
    D = numer/denom
    return D

def delta_D(dr, dt, r, t, w):
    term1 = (r*dr)/(t**2 * w)
    term2 = (r**2 * dt)/(t**3 * w)
    return np.sqrt(term1**2 + term2**2)

print(delta_D(0.1e-3,1,7.8e-3,116.8,np.pi/120))
# %%
#Task2.4
titles = ['thermal_1min_a.txt','thermal_1min_b.txt',\
'thermal_2min_a.txt','thermal_2min_b.txt',\
'thermal_4min_a.txt','thermal_4min_b.txt',\
'thermal_6min.txt', 'thermal_6min.txt',\
'thermal_8min.txt', 'thermal_16min.txt']
for title in titles:
    x1, y1 = np.loadtxt(title, unpack=True, skiprows=3)
    x1 = x1*1e-1
    plt.title(title)
    plt.plot(x1,y1)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.show()
# %%
x1, y1 = np.loadtxt("thermal_8min.txt", unpack=True, skiprows=3)
x1 = x1*1e-1

plt.plot(x1,y1,label='8 min Data')
sq1 = t1.squarewave(60*8,x1[-1],100)
sq1.plot_actual(label='Initial square wave')
sq1.plot(2)
# plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.plot(x1[np.argmax(y1)],y1[np.argmax(y1)],'o')
plt.show()
# %%
period = [120,240,360,480]
D_values = [7.9e-8,8.5e-8,8.6e-8,9.7e-8]
error = [0.3e-8,0.2e-8,0.2e-8,0.3e-8]

plt.errorbar(period, D_values, yerr=error, capsize=5, fmt='.')
plt.xlabel('Period (s)')
plt.ylabel('D (m^2s^-1)')
plt.plot(np.linspace(0,500,1000),0.75e-7*np.ones(1000), c='blue')
plt.grid()
plt.show()
# %%
