from pylab import *
import numpy as np

x = linspace(0,2*pi,129)
x = x[:-1]
y = sin(5*x)
Y = fftshift(fft(y))/128
w = linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $Sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y), 'ro', lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii], angle(Y[ii]), 'go', lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$k$", size=16)
grid(True)
show()

t = linspace(-4*pi,4*pi,513)
t = t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y = fftshift(fft(y))/512
w = linspace(-64,64,513)
w = w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y), 'ro', lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii], angle(Y[ii]), 'go', lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$k$", size=16)
grid(True)
show()

t = linspace(-4*pi,4*pi,513)
t = t[:-1]
y=(sin(t))**3
Y = fftshift(fft(y))/512
w = linspace(-64,64,513)
w = w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $Sin^3(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y), 'ro', lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii], angle(Y[ii]), 'go', lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$k$", size=16)
grid(True)
show()

t = linspace(-4*pi,4*pi,513)
t = t[:-1]
y=(cos(t))**3
Y = fftshift(fft(y))/512
w = linspace(-64,64,513)
w = w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $Cos^3(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y), 'ro', lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii], angle(Y[ii]), 'go', lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$k$", size=16)
grid(True)
show()

t = linspace(-4*pi,4*pi,513)
t = t[:-1]
y=cos(20*t+5*cos(t))
Y = fftshift(fft(y))/512
w = linspace(-64,64,513)
w = w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-35,35])
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $cos(20*t+5*cos(t))$")
grid(True)
subplot(2,1,2)
# plot(w,angle(Y), 'ro', lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii], angle(Y[ii]), 'go', lw=2)
xlim([-35,35])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$k$", size=16)
grid(True)
show()


# X = 2
# length = 8193
# error = 10
# Yold = np.zeros(length-1)
# Ynew = Yold
# while (error>9e-7):
#     # error = np.max(abs(Ynew[::2]-Yold))
#     # length = X*2+1
#     t = linspace(-X,X,length)
#     t = t[:-1]
#     y=exp(-(t**2)/2)
#     Ynew = fftshift(fft(y))/(length-1)
#     error = np.max(abs(Ynew-Yold))
#     Yold = Ynew
    # X = X*2



# print(X)
# print(error)


SET_OF_VALUES = np.array([[129,32],[129,64], [129, 128],[513,32], [513,64], [513,128], [1025,32], [1025,64], [2049,128], [2049,64], [4097,64],[4097,256]])

for a in SET_OF_VALUES:
    N = a[0]
    x = a[1]
    t = linspace(-x,x,N)
    t = t[:-1]
    y=exp(-(t**2)/2)
    Y = fftshift(fft(y))/(N-1)
    w = linspace(-x,x,N)
    w = w[:-1]
    plot(w,abs(Y),lw=2)
    xlim([-40,40])
    ylabel(r"$|y|$", size = 16)
    title(r"Spectrum of $e^{-t^2/2}$")
    plt.legend([f'n={N-1},t={x}'])
    grid(True)
    grid(True)
    show()

N = 4097
x = 80
t = linspace(-x,x,N)
t = t[:-1]
y=exp(-(t**2)/2)
Y = fftshift(fft(y))/(N-1)
w = linspace(-x,x,N)
w = w[:-1]
plot(w,abs(Y),'k',lw=2)
xlim([-15,15])
actual_gaussian = np.sqrt(2*np.pi)*np.exp(-(w**2)/2)
scaling_factor = np.max(abs(actual_gaussian))/np.max(abs(Y))
plt.plot(w,actual_gaussian/scaling_factor,'y',alpha = 0.5, linewidth = 4)
ylabel(r"$|y|$", size = 16)
title(r"Spectrum of $e^{-t^2/2}$")
plt.legend([f'n={N-1},t={x}','Actual value'])
# print(np.max(abs(actual_gaussian/scaling_factor-abs(Y))))
grid(True)
grid(True)
show()
