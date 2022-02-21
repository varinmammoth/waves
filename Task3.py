#%%
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

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

data_fourier.fourier(10)
square_fourier.fourier(10)

y1_1 = data_fourier.get_nth_phase_mode(x1, 1)
y1square_1 = 0
# for i in range (1,10):
#     y1_1 = data_fourier.get_nth_phase_mode(x1, i)
#     y1square_1 = square_fourier.get_nth_phase_mode(x1, i)
#     plt.plot(x1, y1_1, label='x=delta_r')
#     plt.plot(x1, y1square_1, label='x=0')
#     plt.title(i)
#     plt.legend()
#     # plt.ylim(-11,11)
#     plt.show()

# %%
#Check if Fourier class is providing correct results:
y1_1 = 0
for i in range(1,10):
    y1_1 += data_fourier.get_nth_phase_mode(x1, i)
y1_1 += (data_fourier.get_a0()/2)

plt.plot(x1, y1, c='green')
plt.plot(x1, y1_1, c='blue')
plt.show()
# %%
y1_square_1 = 0
for i in range(1,10):
    y1_square_1 += square_fourier.get_nth_phase_mode(x1, i)
    
y1_square_1 += square_fourier.get_a0()/2
plt.plot(x1, y1square, c='green')
plt.plot(x1, y1_square_1)
plt.show()


# %%
