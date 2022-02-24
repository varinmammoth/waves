#%%
from tkinter import Y
import numpy as np
from ray import data
import scipy.signal as sp
import matplotlib.pyplot as plt

def get_D(w, y, r):
    denom = (np.log(y)/r)**2
    denom = 2*denom
    D = w/denom
    return D

def get_D_phaselag(w, r, phi):
    numer = w*(r**2)
    denom = 2*(phi**2)
    D = numer/denom
    return D

def delta_D(dr, dt, r, t, w):
    term1 = (r*dr)/(t**2 * w)
    term2 = (r**2 * dt)/(t**3 * w)
    return np.sqrt(term1**2 + term2**2)

def trap_int(x, y):
    sum = 0
    for i in range(0,len(y)-1):
        sum += y[i]*(x[i+1]-x[i]) - 0.5*(y[i]-y[i+1])*(x[i+1]-x[i])
    return sum

class Fourier:
    def __init__(self, x, y, T):
        self.T = T
        self.x = [] #truncated x to 1 period
        self.y = [] #truncated y to 1 period

        for i in range(0,len(x)):
            if x[i] < self.T:
                self.x.append(x[i])
                self.y.append(y[i])
            else:
                break

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def fourier(self, n_terms):
        a_0 = (2/self.T)*trap_int(self.x,self.y)
        #self.a_n and self.b_n returns coefficients from 0 to n_terms
        self.a_n = [a_0]
        self.b_n = [0]
        for n in range(1, n_terms+1):
            a_n = (2/self.T)*trap_int(self.x,self.y*np.cos((2*np.pi*n/self.T)*self.x))
            b_n = (2/self.T)*trap_int(self.x,self.y*np.sin((2*np.pi*n/self.T)*self.x))
            self.a_n.append(a_n)
            self.b_n.append(b_n)

        self.a_n = np.array(self.a_n)
        self.b_n = np.array(self.b_n)

        #self.beta_n and self.phi_n reuturns coefficients from 0 to n_terms,
        #although the n=0 term is not physical/ mathematically required
        self.beta_n = np.sqrt(np.square(self.a_n) + np.square(self.b_n))
        self.phi_n = -1*np.arctan2(self.a_n, self.b_n)

    def get_fourier_coeff(self):
        return self.a_n, self.b_n

    def get_fourier_phase_coeff(self):
        return self.beta_n, self.phi_n

    def get_nth_phase_mode(self, x, n):
        beta = self.beta_n[n]
        phi = self.phi_n[n]
        a0 = self.a_n[0]
        return beta*np.sin((2*np.pi*n/self.T)*x - phi)
    
    def get_nth_mode(self, x, n):
        a = self.a_n[n]
        b = self.b_n[n]
        a0 = self.a_n[0]
        return a*np.cos((2*np.pi*n/self.T)*x) + b*np.sin((2*np.pi*n/self.T)*x)

    def get_a0(self):
        return self.a_n[0]

def square(t, A, T):
    return (A/2)*sp.square(2 * np.pi * t / T) + A/2
# %%
T = 240

x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True, skiprows=3)
x1 = x1*1e-1
y1square = square(x1, 100, T)

data_fourier = Fourier(x1, y1, T)
square_fourier = Fourier(x1, y1square, T)

data_fourier.fourier(15)
square_fourier.fourier(15)
# %%
#Check if Fourier class is providing correct results:
y1_1 = 0
for i in range(1,10):
    y1_1 += data_fourier.get_nth_phase_mode(x1, i)
y1_1 += (data_fourier.get_a0()/2)

plt.plot(x1, y1, c='green', label='Actual')
plt.plot(x1, y1_1, label='Fourier')
plt.legend(loc=1)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.show()

y1_square_1 = 0
for i in range(1,10):
    y1_square_1 += square_fourier.get_nth_phase_mode(x1, i)
    
y1_square_1 += square_fourier.get_a0()/2
plt.plot(x1, y1square, c='green', label='Actual')
plt.plot(x1, y1_square_1, label='Fourier')
plt.legend(loc=1)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.show()
# %%
for i in range(1,6):
    plt.plot(x1, square_fourier.get_nth_phase_mode(x1, i), label='x=0')
    plt.plot(x1, data_fourier.get_nth_phase_mode(x1, i), label='x=delta_r')
    plt.legend(loc=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.title(i)
    plt.show()

#%%
def Dtf_error(w, r, y, dr, tin, tout):
    dT=0.000001
    dy = np.sqrt((dT/tout)**2 + (dT*tin/(tout**2))**2)
    error = np.sqrt((w*r*dr/(np.log(y)**2))**2 + (w*dy*(r**2))/((np.log(y)**3)*y)**2)
    return error

def DPl_error(w, r, phi, dr, beta):
    dT=0.000001
    dphi = dT/beta
    error = np.sqrt((w*r*dr/(phi**2))**2 + (w*(r**2)*dphi/phi)**2)
    return error

#Analysis
D_pl_all = []
D_pl_all_err = []
D_tf_all = []
D_tf_all_err = []

titles = ['thermal_1min_a.txt',\
'thermal_2min_a.txt',\
'thermal_4min_a.txt',\
'thermal_6min.txt', \
'thermal_8min.txt']
periods = [60, 120, 240, 360, 480]
for i in range(0,len(titles)):
    T = periods[i]
    x1, y1 = np.loadtxt(titles[i], unpack=True, skiprows=3)
    x1 = x1*1e-1
    y1square = square(x1, 100, T)

    nterms = 7
    data_fourier = Fourier(x1, y1, T)
    square_fourier = Fourier(x1, y1square, T)
    data_fourier.fourier(nterms)
    square_fourier.fourier(nterms)

    y1_1 = 0
    for i in range(1,nterms):
        y1_1 += data_fourier.get_nth_phase_mode(x1, i)
    y1_1 += data_fourier.get_a0()/2
    y1_square_1 = 0
    for i in range(1,nterms):
        y1_square_1 += square_fourier.get_nth_phase_mode(x1, i)   
    y1_square_1 += square_fourier.get_a0()/2
    # plt.plot(x1, y1square, label='Actual')
    # plt.plot(x1,y1_square_1, label='Fourier')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Temperature (C)')
    # plt.legend(loc=1)
    # plt.show()
    # plt.plot(x1, y1, label='Actual')
    # plt.plot(x1,y1_1, label='Fourier')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Temperature (C)')
    # plt.legend(loc=1)
    # plt.show()

    #get the amplitude and phase for each term
    amp_outter, phi_outter = square_fourier.get_fourier_phase_coeff()
    amp_inner, phi_inner = data_fourier.get_fourier_phase_coeff()

    #find phase difference and transmission
    delta_phi = phi_outter - phi_inner
    transmission = np.divide(amp_inner, amp_outter)

    #we know the even terms are non-physical and due to random noise
    #so keep only odd terms
    delta_phi = delta_phi[1::2]
    transmission = transmission[1::2]
    delta_phi = np.array([abs(i) for i in delta_phi])

    n = np.arange(1, nterms+1)[0::2]
    w = (2*np.pi/T)*n

    D_pl = get_D_phaselag(w, 7.8e-3, delta_phi)
    D_pl_err = DPl_error(w, 7.8e-3,delta_phi,0.1e-3,amp_inner[1::2])
    D_trans = get_D(w,transmission,7.8e-3)
    D_trans_err = Dtf_error(w,7.8e-3,transmission,0.1e-3,amp_inner[1::2],amp_outter[1::2])

    D_pl_all.append(D_pl)
    D_pl_all_err.append(D_pl_err)
    D_tf_all.append(D_trans)
    D_tf_all_err.append(D_trans_err)

    # plt.title(str(T)+'s'+' Transmission factor')
    # plt.errorbar(n, D_trans, yerr=D_trans_err, fmt='.', capsize=5)
    # plt.xlabel('n')
    # plt.ylabel('D ' + r'$(m^{2} s^{-1})$')
    # plt.show()
    # plt.title(str(T)+'s'+' Phase lag')
    # plt.errorbar(n, D_pl, yerr=D_pl_err, fmt='.', capsize=5)
    # plt.xlabel('n')
    # plt.ylabel('D ' + r'$(m^{2} s^{-1})$')
    # plt.show()


# %%
import matplotlib.patches as mpatches
for n in range(0, len(np.arange(1, nterms+1)[0::2])): 
    for i in range(0,len(periods)):
        plt.title('Mode: ' + str(2*n+1))
        plt.errorbar(periods[i], D_pl_all[i][n], yerr=D_pl_all_err[i][n], fmt='.', capsize=2, c='green')
        plt.errorbar(periods[i], D_tf_all[i][n], yerr=D_tf_all_err[i][n], fmt='.', capsize=2, c='red')
        plt.legend()
        plt.xlabel('Period (s)')
        plt.ylabel('D ' + r'$(m^{2} s^{-1})$')
    red_patch = mpatches.Patch(color='red', label=r'$D_{TF}$')
    green_patch = mpatches.Patch(color='green', label=r'$D_{PL}$')
    plt.legend(handles=[red_patch, green_patch])
    plt.grid()
    plt.show()
        
# %%
