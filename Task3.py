#%%
from cmath import pi
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
    #error = np.sqrt((w*r*dr/(np.log(y)**2))**2 + (w*dy*(r**2))/((np.log(y)**3)*y)**2)
    error = np.sqrt((w*r*dr/(np.log(y)**2))**2)
    return error

def DPl_error(w, r, phi, dr, beta):
    dT=0.000001
    dphi = dT/beta
    #error = np.sqrt((w*r*dr/(phi**2))**2 + (w*(r**2)*dphi/phi)**2)
    error = np.sqrt((w*r*dr/(phi**2))**2)
    return error

#Analysis
D_pl_all = []
D_pl_all_err = []
D_tf_all = []
D_tf_all_err = []
trans_all = []
pl_all = [] 

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
    for i in range(0,len(delta_phi)):
        if i == 0:
            while delta_phi[i] < 0:
                delta_phi[i] += 2*np.pi
        else:
            while delta_phi[i] < delta_phi[i-1]:
                delta_phi[i] += np.pi*2
    #delta_phi = np.array([abs(i) for i in delta_phi])

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
    trans_all.append(transmission)
    pl_all.append(delta_phi)

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
D_pl_all_b = []
D_pl_all_b_err = []
D_tf_all_b = []
D_tf_all_b_err = []

periods = [60, 120, 240, 360, 480]
for i in range(0,len(periods)):

    trans_data = trans_all[i]#input your transmission data here
    trans_periods = [periods[i]*j for j in [1,1,1,1]] #input the corresponding periods here (s)
    phase_data = pl_all[i] #input your phase lag data here
    phase_periods = [periods[i]*j for j in [1,1,1,1]]#input the corresponding periods here (s)
    r_in = 0.0025 #inner  radius (m)
    r_out = r_in + 7.88e-3 #outer radius (m)
    no_terms = 100 #number of terms you want to truncate the J0 Bessel Series to

    # =============================================================================   
    # BESSEL TRANSMISSION AND BESSEL PHASE LAG FUNCTIONS
    # =============================================================================
    # General comments:
    # -----------------
    #   Terms in the Bessel Series are complex. They take the form:
    #       -  J0 = re_in + i(im_in) ,  at r_in;
    #       -  J0 = re_out + i(im_out) ,  at r_out.
    #   We consider J0 as a function of alpha, where alpha = sqrt(w/D)*r_in.
    #
    #   The functions are designed to be used for fitting the transmission factors
    #   and phase lags obtained in the Fourier analysis of the experimental data.
    #    
    #   A Secant root-finding method is used to find the alpha than matches
    #   the Bessel phase lag / transmision values to the Fourier ones. 
    #        
    #   Finally, the diffusivity is obtained from alpha.
    #     
    # About the iterator I:
    # --------------------
    #   The Bessel series is obtained via an iterative process. 
    #   Each iteration corresponds to 2 terms in the series, so upper limit is set 
    #   to be I < (no_terms/2).
    # 
    # =============================================================================


    from scipy.optimize import newton
    import numpy as np

    def bessel_trans_fit(alpha): 
        """
        FUNCTION THAT CALCULATES TRANSMISSION FACTORS VIA BESSEL ANALYSIS 
        
        It computes the J0 at the inner radius (r_in), the outer radius (r_out), 
        and the transmission factor J0_in/J0_out (bessel_trans).
        
        Parameters:
        ----------
        alpha = sqrt(w/D)*r_in      
        
        Returns:
        -------
        For a given Fourier transmission value (fourier_trans), bessel_trans_fit 
        returns:
                bessel_trans - fourier_trans   
        Later, a secant root-finding method is used to find alpha when
        bessel_trans - fourier_trans = 0. 
        """
        I = 0 
        ratio = r_out/r_in
        re_in = re_out = im_in = im_out = 0
        
        #TO UPDATE VARIABLE PART OF TERMS IN SERIES:
        term_in = 1  #r_in series
        term_out = 1 #r_out series
        
        #FOR UPDATING CONSTANT COEFFICIENT OF TERMS IN SERIES:
        re_coeff = 1  #real part
        im_coeff = -re_coeff / 4 #imaginary part

        while I<(no_terms/2):
            #updating r_in series:
            re_in = re_in + re_coeff * term_in
            term_in = term_in * alpha**2
            im_in = im_in + im_coeff * term_in
            term_in = term_in * alpha**2 
            
            #updating r_out series:
            re_out = re_out + re_coeff * term_out
            term_out = term_out * (alpha * ratio)**2 
            im_out = im_out + im_coeff * term_out
            term_out = term_out * (alpha * ratio)**2
            
            #updating coeffs and iterator:
            re_coeff = -re_coeff / (4*(2*I + 1)*(2*I + 2))**2
            I = I + 1 
            im_coeff = -re_coeff / (4*(2*I + 1)**2)
            
        #transmission factor:
        bessel_trans = np.sqrt((re_in**2 + im_in**2) / (re_out**2 + im_out**2))
        
        return bessel_trans - fourier_trans


    def bessel_phase_fit(alpha):
        """
        FUNCTION THAT CALCULATES PHASE LAGS VIA BESSEL ANALYSIS 
        
        It computes the J0 at the inner radius (r_in), the outer radius (r_out), 
        and the phase lag, phase_in - phase_out (bessel_phase_lag).
        
        Parameters:
        ----------
        alpha = sqrt(w/D)*r_in      
        
        Returns:
        -------
        For a given Fourier phase lag value (fourier_phase_lag),bessel_phase_fit 
        returns:
                bessel_phase_lag - fourier_phase_lag 
        Later, a secant root-finding method is used to find alpha when
        bessel_phase_lag - fourier_phase_lag = 0. 
        """
        I = 0 
        ratio = r_out/r_in
        re_in = re_out = im_in = im_out = 0
        
        #TO UPDATE VARIABLE PART OF TERMS IN SERIES:
        term_in = 1  #r_in series
        term_out = 1 #r_out series
        
        #FOR UPDATING CONSTANT COEFFICIENT OF TERMS IN SERIES:
        re_coeff = 1  #real part
        im_coeff = -re_coeff / 4 #imaginary part

        while I<(no_terms/2):
            #updating r_in series:
            re_in = re_in + re_coeff * term_in
            term_in = term_in * alpha**2
            im_in = im_in + im_coeff * term_in
            term_in = term_in * alpha**2 
            
            #updating r_out series:
            re_out = re_out + re_coeff * term_out
            term_out = term_out * (alpha * ratio)**2 
            im_out = im_out + im_coeff * term_out
            term_out = term_out * (alpha * ratio)**2
            
            #updating coeffs and iterator:
            re_coeff = -re_coeff / (4*(2*I + 1)*(2*I + 2))**2
            I = I + 1 
            im_coeff = -re_coeff / (4*(2*I + 1)**2)
            
        phase_out = np.arctan2(im_out,re_out)
        phase_in = np.arctan2(im_in,re_in)
        
        if phase_in > phase_out:
            bessel_phase_lag = phase_in - phase_out 
        else:
            bessel_phase_lag = phase_in - phase_out + 2*np.pi 
        
        return bessel_phase_lag - fourier_phase_lag



    # =============================================================================
    # FITTING FUNCTIONS TO EXPERIMENTAL DATA AND CALCULATING DIFFUSIVITY:
    # =============================================================================

    #FITTING bessel_trans_fit TO TRANSMISSION DATA 
    trans_alphas = []
    for trans in trans_data:
        fourier_trans = trans
        b = newton(bessel_trans_fit,x0=1) #secant method 
        trans_alphas.append(b)

    #FITTING bessel_phase_fit TO PHASE DATA 
    phase_alphas = []
    for phase in phase_data:
        fourier_phase_lag = phase
        b = newton(bessel_phase_fit,x0=1) #secant method
        phase_alphas.append(b)

    #CALCULATING PHASE DIFFUSIVITY FROM ALPHA
    D_phase = []
    for i in range(len(phase_alphas)):
        w = (2*np.pi/phase_periods[i])
        alpha = phase_alphas[i]
        D_p = (w*r_in**2)/(alpha**2)
        D_phase.append(D_p)
        
    #CALCULATING TRANSMISSION DIFFUSIVITY FROM ALPHA
    D_trans = []
    for i in range(len(trans_alphas)):
        w = (2*np.pi/trans_periods[i])
        alpha = trans_alphas[i]
        D_t = (w*r_in**2)/(alpha**2)
        D_trans.append(D_t)


    # print("\nDIFFUSIVITY - BESSEL ANALYSIS")
    # print('\nDiffusivity - Transmission Factors:')
    # #print(D_trans)
    # print('\nDiffusivity - Phase Lags:')
    # #print(D_phase)
    D_pl_all_b.append(D_phase)
    D_tf_all_b.append(D_trans)
    
    percent_err = np.divide(np.array(D_pl_all_err[i]), np.array(D_pl_all[i]))
    err = np.multiply(percent_err, np.array(D_phase))
    D_pl_all_b_err.append(err)
    percent_err = np.divide(np.array(D_tf_all_err[i]), np.array(D_tf_all[i]))
    err = np.multiply(percent_err, np.array(D_trans))
    D_tf_all_b_err.append(err)
# %%
#For the 4 minute data set:
plt.errorbar(np.arange(1, nterms+1)[0::2], D_pl_all_b[2], yerr=D_pl_all_b_err[2], fmt='.', capsize=3, label='Bessel '+r'$D_{PL}$')
plt.errorbar(np.arange(1, nterms+1)[0::2], D_tf_all_b[2], yerr=D_tf_all_b_err[2], fmt='.', capsize=3, label='Bessel '+r'$D_{TF}$')
plt.errorbar(np.arange(1, nterms+1)[0::2], D_pl_all[2], yerr=D_pl_all_err[2], fmt='.', capsize=3, label='Fourier '+r'$D_{PL}$')
plt.errorbar(np.arange(1, nterms+1)[0::2], D_tf_all[2], yerr=D_tf_all_err[2], fmt='.', capsize=3, label='Fourier '+r'$D_{TF}$')
plt.xlabel('n')
plt.ylabel('D ' + r'$(m^{2} s^{-1})$')
plt.legend()
plt.grid()
plt.show()
# %%
