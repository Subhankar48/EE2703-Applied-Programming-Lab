import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.signal import convolve2d
from numpy import sin,cos
from numpy import pi
from pylab import ceil,log2

filter_coeffs = np.asanyarray(np.loadtxt('h.csv').reshape(1,-1))[0].reshape(-1,1)
w,h = sp.freqz(filter_coeffs,whole=True)
plt.subplot(2,1,1)
plt.plot(w,abs(h))
plt.title('Magnitude')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(h))
plt.title('Phase')
plt.grid(True)
plt.show()

n = np.arange(1025)
x = np.asanyarray(cos(0.2*pi*n)+cos(0.85*pi*n), dtype = np.complex).reshape(-1,1)
plt.plot(n[:110],np.real(x[:110]))
plt.title('x=$cos(0.2πn)+cos(0.85πn)$')
plt.xlabel('n')
plt.ylabel('x')
plt.grid(True)
plt.show()

y = convolve2d(x,filter_coeffs, mode='valid')
plt.plot(np.arange(len(y))[0:200], np.real(y[0:200]))
plt.grid(True)
plt.xlabel('n')
plt.ylabel('y')
plt.title('$y=x*h$')
plt.show()


Y=np.concatenate((filter_coeffs,np.zeros((len(x)-len(filter_coeffs),1))))
temp2 = Y[:,0]
temp1 = x[:,0]
y1=np.fft.ifft(np.fft.fft(temp1)*np.fft.fft(temp2))
plt.plot(np.arange(len(y1))[0:200],np.real(y1)[:200])
plt.xlabel('n')
plt.ylabel('y1')
plt.grid(True)
plt.title('circular convolution')
plt.show()


def circular_conv(x,h):
    P = len(h)
    n_ = int(ceil(log2(P)))
    h_ = np.concatenate((h,np.zeros(int(2**n_)-P)))
    P = len(h_)
    n1 = int(ceil(len(x)/2**n_))
    x_ = np.concatenate((x,np.zeros(n1*(int(2**n_))-len(x))))
    y = np.zeros(len(x_)+len(h_)-1)
    for i in range(n1):
        temp = np.concatenate((x_[i*P:(i+1)*P],np.zeros(P-1)))
        y[i*P:(i+1)*P+P-1] += np.fft.ifft(np.fft.fft(temp)*np.fft.fft(np.concatenate((h_,np.zeros(len(temp)-len(h_)))))).real
    return y

a = circular_conv(x[:,0], filter_coeffs[:,0])
plt.plot(np.arange(len(a))[:200],a[:200])
plt.grid(True)
plt.title('Linear convolution using circular convolution')
plt.xlabel('n')
plt.show()

f = open('x1.csv','r')
raw_data =f.read().splitlines()
for p in range(len(raw_data)):
    raw_data[p] = complex(raw_data[p].replace('i','j'))

ZADOFF_CHU_COEFFICIENTS = np.asanyarray(raw_data,dtype=np.complex).reshape(-1,1)
ZADOFF_CHU_COEFFICIENTS_delayed = np.roll(ZADOFF_CHU_COEFFICIENTS,5)
y2 = np.fft.ifft((np.fft.fft((np.roll(ZADOFF_CHU_COEFFICIENTS[:,0],5)))*np.conj(np.fft.fft(ZADOFF_CHU_COEFFICIENTS[:,0]))))
plt.stem(abs(y2), use_line_collection=True)
plt.xlim([0,10])
plt.grid(True)
plt.title("Auto-correlation of a delayed sample of the Zadoff-Chu signal")
plt.show()
# print(len(ZADOFF_CHU_COEFFICIENTS))