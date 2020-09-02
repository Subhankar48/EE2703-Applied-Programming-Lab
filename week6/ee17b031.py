#libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.signal import convolve2d
import time
from scipy.linalg import lstsq
import sys

t = np.linspace(0,50,int(5e3))
f1_t = np.cos(1.5*t)*np.exp(-0.5*t)
f2_t = np.cos(1.5*t)*np.exp(-0.05*t)

def plotter(f,t):
    transfer_function = sp.lti([1],[1,0,2.25])
    t,h_t = sp.impulse(transfer_function,None,t)
    t,y,svec = sp.lsim(transfer_function, f, t)
    plt.subplot(3,1,1)
    plt.plot(t,f, 'r')
    plt.grid(True)
    plt.title("Input")
    plt.subplot(3,1,2)
    plt.plot(t, h_t, 'y')
    plt.grid(True)
    plt.title('Impulse response')
    plt.subplot(3,1,3)
    plt.plot(t,y,'b')
    plt.title('Output')
    plt.grid(True)
    plt.show()

plotter(f1_t,t)
plotter(f2_t,t)

t = np.linspace(0,100,int(1e4))
x_num = np.poly1d([1])
x_den = np.poly1d([1,0,2.25])
transfer_function_temp = sp.lti(x_num,x_den)
w_vals = np.arange(1.4, 1.62, 0.05)
count = 1
for w in w_vals:
    F_num = np.poly1d([1,0.05])
    f_t = np.cos(w*t)*np.exp(-0.05*t)
    t,y_temp,svec = sp.lsim(transfer_function_temp, f_t, t)
    plt.subplot(3,2,count)
    temp_string = f"$ω$= {w}"
    plt.title(temp_string)
    plt.plot(t,y_temp)
    plt.grid(True)
    count = count+1
plt.show()


#q4
time = np.linspace(0,20,int(5e3))
X = sp.lti([1,0,2],[1,0,3,0])
Y = sp.lti([2],[1,0,3,0])
time, x = sp.impulse(X, None, time)
time, y = sp.impulse(Y, None, time)
plt.plot(time,x,'y')
plt.title('x(t)')
plt.grid(True)
plt.plot(time,y)
plt.title('plot of x(t) and y(t)')
plt.legend(['x(t)', 'y(t)'])
plt.show()

#q5
RLC_transfer_function = sp.lti([1e12], [1,1e8,1e12])
w,S,phi = RLC_transfer_function.bode()
plt.subplot(2,1,1)
plt.semilogx(w,S)
plt.title('Magnitude Response')
plt.grid(True)
plt.ylabel('$|H(Ω)|$')
plt.subplot(2,1,2)
plt.semilogx(w,phi)
plt.title("Phase Response")
plt.ylabel("$φH(Ω)$")
plt.grid(True)
plt.show()

#q6
time_vec = np.linspace(0,30e-6, int(1e5))
v_i = np.cos(1000*time_vec)-np.cos(1e6*time_vec)
time_vec,v_o, svec = sp.lsim(RLC_transfer_function,v_i,time_vec)
plt.plot(time_vec, v_o)
plt.title('$v_{o}(t)$ upto 30us')
plt.xlabel('t')
plt.grid(True)
plt.show()

time_vec = np.linspace(0,10e-3, int(1e5))
v_i = np.cos(1000*time_vec)-np.cos(1e6*time_vec)
time_vec,v_o, svec = sp.lsim(RLC_transfer_function,v_i,time_vec)
plt.plot(time_vec, v_o)
plt.title('$v_{o}(t)$')
plt.xlabel('t')
plt.grid(True)
plt.show()
