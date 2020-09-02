import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.signal import convolve2d
# from numpy import sin,cos
# from numpy import pi

from pylab import *



'''Examples'''

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=sin(sqrt(2)*t)
y[0]=0
y=fftshift(y)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-1.png")
show()

t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
# y=sin(sqrt(2)*t)
figure(2)
plot(t1,sin(sqrt(2)*t1),'b',lw= 2)
plot(t2,sin(sqrt(2)*t2),'r',lw=2)
plot(t3,sin(sqrt(2)*t3),'r',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid(True)
savefig("fig10-2.png")
show()

from pylab import *
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
y=sin(sqrt(2)*t1)
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
grid(True)
savefig("fig10-3.png")
show()

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=t
y[0]=0
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure()
semilogx(abs(w),20*log10(abs(Y)),lw=2)
xlim([1,10])
ylim([-20,0])
xticks([1,2,5,10],["1","2","5","10"],size=16)
ylabel(r"$|Y|$ (dB)",size=16)
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-4.png")
show()

n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t1)*wnd
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid(True)
savefig("fig10-5.png")
show()

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t)*wnd
y[0]=0
y=fftshift(y) 
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-6.png")
show()

t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/255))
y=sin(sqrt(2)*t)
y=y*wnd
y[0]=0 
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-7.png")
show()


'''Question 2'''
#without hamming window
t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
y=cos(0.86*t)**3
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(0.86t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-8.png")
show()

#with hamming window
t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
y=cos(0.86*t)**3
wnd=fftshift(0.54+0.46*cos(2*pi*n/255))
y=y*wnd
y=fftshift(y) 
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(0.86t)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-9.png")
show()

'''Question 3'''
t=linspace(-pi,pi,129);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(128)
Omega = np.random.normal(loc=1.25,scale=0.08)
if Omega>=1.5:
    Omega=1.5
elif Omega<0.5:
    Omega = 0.5

delta = np.random.uniform(low=-pi,high=pi)
print(f"real ω={Omega}")
print(f"real δ ={delta}")
y=cos(Omega*t+delta)
y1 = cos(0.5*t)
wnd=fftshift(0.54+0.46*cos(2*pi*n/127))
y=y*wnd
y1=y1*wnd
y=fftshift(y)
y1 = fftshift(y1)
Y1 = fftshift(fft(y1))/128 
Y=fftshift(fft(y))/128
w=linspace(-pi*fmax,pi*fmax,129);w=w[:-1]
figure()
subplot(2,1,1)
# plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
plot(w,abs(Y))
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos(ωt+δ)\times w(t)$")
grid(True)
subplot(2,1,2)
i1 = (abs(Y)>0.01)
i2 = (abs(Y1)>0.01)
plot(w[i1],angle(Y[i1]),'ro',lw=2)
plot(w[i2],angle(Y1[i2]),'yo',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-13.png")
i1 = (abs(Y)>0.01)&(w>0)
i2 = (abs(Y1)>0.01)&(w>0)
N_max = np.argmax(abs(Y))
predicted_delta = -angle(Y[N_max])

# predicted_delta =sum(angle(Y[i1]))/2
# print(angle(Y[i1]))
show()
# ii= where(w>0)
a = sum(abs(Y)**2.1*abs(w))/sum(abs(Y)**2.1)
print(f"predicted ω={a}")
print(f"predicted δ={predicted_delta}")
# show()



print('***************************\n***************************')


'Question 4'

t=linspace(-pi,pi,129);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(128)
Omega = np.random.normal(loc=1.25,scale=0.08)
if Omega>=1.5:
    Omega=1.5
elif Omega<0.5:
    Omega = 0.5

delta = np.random.uniform(low=-pi,high=pi)
print(f"real ω={Omega}")
print(f"real δ ={delta}")
y=cos(Omega*t+delta)+0.1*randn(len(n))
y1 = cos(0.5*t)

wnd=fftshift(0.54+0.46*cos(2*pi*n/127))
y=y*wnd
y1=y1*wnd
y=fftshift(y)
y1 = fftshift(y1)
Y1 = fftshift(fft(y1))/128 
Y=fftshift(fft(y))/128
w=linspace(-pi*fmax,pi*fmax,129);w=w[:-1]
figure()
subplot(2,1,1)
# plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
plot(w,abs(Y))
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of noisy $\cos(ωt+δ)\times w(t)$")
grid(True)
subplot(2,1,2)
i1 = (abs(Y)>0.01)
i2 = (abs(Y1)>0.01)
plot(w[i1],angle(Y[i1]),'ro',lw=2)
plot(w[i2],angle(Y1[i2]),'yo',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-13.png")
i1 = (abs(Y)>0.01)&(w>0)
i2 = (abs(Y1)>0.01)&(w>0)
predicted_delta =sum(angle(Y[i1]))/2

N_max = np.argmax(abs(Y))
predicted_delta = -angle(Y[N_max])

# print(angle(Y[i1]))
show()
# ii= where(w>0)
a = sum(abs(Y)**(2.8)*abs(w))/sum(abs(Y)**2.8)
print(f"predicted noisy ω={a}")
print(f"predicted noisy δ={predicted_delta}")



'Question 5'
#without hamming window
t=linspace(-pi,pi,1025);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=cos(t*16*(1.5+t/(2*pi)))
y=fftshift(y)
Y=fftshift(fft(y))/1024
w=linspace(-pi*fmax,pi*fmax,1025);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-50,50])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos(16(1.5+t/2{\pi})t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-50,50])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-10.png")
show()

#with hamming window
t=linspace(-pi,pi,1025);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=cos(t*16*(1.5+t/(2*pi)))
# y[0]=0
n=arange(1024)
wnd=fftshift(0.54+0.46*cos(2*pi*n/1023))
y=y*wnd
y=fftshift(y)
Y=fftshift(fft(y))/1024
w=linspace(-pi*fmax,pi*fmax,1025);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-50,50])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos(16(1.5+t/2{\pi})t)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-50,50])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig10-11.png")
show()

'''QUestion 7'''
t=linspace(-pi,pi,1025);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=cos(t*16*(1.5+t/(2*pi)))
plt.plot(t,y)
plt.grid(True)
plt.xlabel('n')
plt.title('Plot of the chirped signal')
plt.show()
y_full = np.zeros((16,64), dtype = np.complex)
for k in range(16):
    y_temp = y[64*k:64*(k+1)]
    # y_temp_fft = np.fft.fft
    y_temp=fftshift(y_temp)
    Y_temp=fftshift(fft(y_temp))/64
    y_full[k] = Y_temp

# print(y_full)

w = linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
N = np.arange(64)
t1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
import mpl_toolkits.mplot3d.axes3d as p3
# plt.contour(abs(y_full))
# plt.show()
# ax=p3.Axes3D(figure())
t1,N = meshgrid(t1,N) 
# surf = ax.plot_surface(t1,N,abs(y_full).T,rstride=1,cstride=1)
# ylabel(r'$N\rightarrow$',size=16)
# xlabel(r'$t\rightarrow$',size=16)
# ax.set_zlabel(r'$|Y|$')
# show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
temp = ax.plot_surface(t1,N,abs(y_full).T,rstride = 1, cstride = 1,alpha = 0.7, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_xlabel(r'$t\rightarrow$',size=16)
ax.set_ylabel(r'$N\rightarrow$',size=16)
plt.title('Chirped Signal spectrum')
fig.colorbar(temp, shrink=0.5, aspect=5)
# ax.set_zlabel('potential', fontsize = 20, rotation = 60)
plt.show()