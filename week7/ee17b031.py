#importing modules
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pylab as p
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy import sin,cos
from numpy import pi

time = np.linspace(0,50e-4,int(5e4))

def lowpass(R1, R2, C1, C2, G, Vi):
    s = Symbol('s')
    A = Matrix([[0,0,1,-1/G], [-1/(1+s*R2*C2), 1,0,0], [0,-G, G,1], [-1/R1-1/R2-s*C1, 1/R2,0,s*C1]])
    b = Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

#A trial of the lowpass filter mentioned
s = Symbol('s')
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
w=p.logspace(0,8,801)
ss=1j*w
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2)
A,b,V2=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
V02 = V2[3]
hf1=lambdify(s,V02,'numpy')
v2=hf1(ss)
p.loglog(w,abs(v2),lw=2)
p.legend(['Impulse response', 'Step response'])
p.title("Magnitude plots of the Low Pass Filter")
p.grid(True)
p.show()
print('G=1000')
print(f"The impulse response of the low pass filter is \n {simplify(Vo)}")
print(f"The step response of the low pass filter is \n {simplify(V02)}")

#generating the transfer function 
numerators_impulse_response_lowpass = np.asanyarray(Poly(fraction(simplify(Vo))[0],s).all_coeffs(), dtype=np.float64)
denominators_impulse_response_lowpass = np.asanyarray(Poly(fraction(simplify(Vo))[1],s).all_coeffs(), dtype=np.float64)



#passing v_{i}(t) through the low pass filter
H_lowpass_impulse = sp.lti(numerators_impulse_response_lowpass, denominators_impulse_response_lowpass)
# w,S, phi = H_lowpass_impulse.bode()
# plt.subplot(2,1,1)
# plt.plot(w, S)
# plt.subplot(2,1,2)
# plt.plot(w, phi)
# plt.show()

low_freq_component = sin(2000*pi*time)
high_freq_component = cos(2*pi*1e6*time)
v_i =low_freq_component + high_freq_component
time,y,svec=sp.lsim(H_lowpass_impulse,v_i,time)

#plotting values for v_i(t) through the low pass filter
plt.subplot(2,2,1)
plt.plot(time,low_freq_component,'r')
plt.grid(True)
plt.title("low frequency component of $v_{i}(t)$")
# plt.show()
plt.subplot(2,2,2)
plt.plot(time[:200], high_freq_component[:200],'b')
plt.title("high frequency component of $v_{i}(t)$")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,10))
plt.grid(True)
# plt.show()
plt.subplot(2,2,3)
plt.plot(time, v_i,'g', alpha = 0.5)
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,10))
# plt.title("$v_{i}(t)$")
plt.grid(True)
plt.xlabel('time')
# plt.show()    
plt.subplot(2,2,4)
plt.plot(time,y,'k')
plt.plot(time, v_i,'y', alpha = 0.5)
plt.legend(['$v_{i}(t)$', '$v_{i}(t)$ through filter    '])
plt.title("$v_{i}(t)$ after the low pass filter")
plt.xlabel('time')
plt.grid(True)
plt.show()

#defining the highpass filter
def highpass(c1,c2,r1,r2, G,Vi):
    s = Symbol('s')
    A = Matrix([[0,0,1,-1/G], [-1/(1+1/s*c2*r2), 1,0,0], [0,-G, G,1], [-s*c1-s*c2-1/r1, s*c2,0,1/r1]])
    b = Matrix([0,0,0,-Vi*s*c1])
    V = A.inv()*b
    return (A,b,V)

A,b,V = highpass(1e-9,1e-9, 1e4, 1e4,1.586,1)
Vo=V[3]
w=p.logspace(0,8,801)
ss=1j*w
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2)
A,b,V2 = highpass(1e-9,1e-9, 1e4, 1e4,1.586,1/s)
V02 = V2[3]
hf1=lambdify(s,V02,'numpy')
v2=hf1(ss)
p.loglog(w,abs(v2),lw=2)
p.legend(['Impulse response', 'Step response'])
p.title("Magnitude plots of the High Pass Filter")
p.grid(True)
p.show()
print('G=1000')
print(f"The impulse response of the high pass filter is \n {simplify(Vo)}")
print(f"The step response of the high pass filter is \n {simplify(V02)}")

#generating the transfer function
numerators_impulse_response_highpass = np.asanyarray(Poly(fraction(simplify(Vo))[0],s).all_coeffs(), dtype=np.float64)
denominators_impulse_response_highpass = np.asanyarray(Poly(fraction(simplify(Vo))[1],s).all_coeffs(), dtype=np.float64)
#passing v_{i}(t) through the low pass filter

H_highpass_impulse = sp.lti(numerators_impulse_response_highpass, denominators_impulse_response_highpass)
time,y,svec=sp.lsim(H_highpass_impulse,v_i,time)

#plotting values for v_i(t) through the high pass filter
plt.subplot(2,2,1)
plt.plot(time,low_freq_component,'r')
plt.grid(True)
plt.title("low frequency component of $v_{i}(t)$")
plt.subplot(2,2,2)
plt.plot(time[:200], high_freq_component[:200],'b')
plt.title("high frequency component of $v_{i}(t)$")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,10))
plt.grid(True)
plt.subplot(2,2,3)
plt.plot(time, v_i,'g', alpha = 0.5)
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,10))
plt.title("$v_{i}(t)$")
plt.grid(True)
plt.xlabel('time')
plt.subplot(2,2,4)
# plt.plot(time[:200], v_i[:200],'y', alpha = 0.5)
plt.plot(time[:200],y[:200],'k')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,10))
plt.title("$v_{i}(t)$ after the high pass filter")
plt.xlabel('time')
plt.grid(True)
plt.show()


#damped oscillation
#first damped oscillation
t1 = np.linspace(0,50e-5,int(1e5))
v_damped1 = np.exp(-5000*t1)*sin(1e6*t1)
t1,y1,svec=sp.lsim(H_highpass_impulse,v_damped1,t1)
# plt.subplot(2,1,1)
plt.plot(t1, v_damped1, 'y')
plt.plot(t1 , y1,'k')
plt.grid(True)

plt.title("$v_{i}(t)=e^{-5000t}sin(1000000t)$")
plt.legend(['$v_{i}(t)$', 'Output = $v_{o}(t)$'])
#The second damped oscillation
plt.show()
# plt.subplot(2,1,2)
t2 = np.linspace(0,50e-2,int(1e5))
v_damped2 = np.exp(-5*t2)*sin(1e3*t2)
t2,y2,svec=sp.lsim(H_highpass_impulse,v_damped2,t2)
plt.plot(t2,v_damped2, 'y' )
plt.plot(t2, y2, 'k')
plt.grid(True)
plt.title("$v_{i}(t)=e^{-5t}sin(1000t)$")
plt.legend(['$v_{i}(t)$', 'Output = $v_{o}(t)$'])
plt.show()

#time domain step response
time = np.arange(100)
step_function = np.ones_like(time)
step_function[0]=0
step_function[1]=0.5


time,y,svec=sp.lsim(H_highpass_impulse,step_function,time)
time,y1,svec=sp.lsim(H_lowpass_impulse,step_function,time)
plt.plot(time, y1, 'k')
plt.plot(time,y, 'r')
plt.plot(time, step_function, 'b')
plt.title("step response of the filters in time domain")
plt.grid(True)
plt.legend(['lowpass filter','highpass filter','$u(t)$'])
plt.show()
