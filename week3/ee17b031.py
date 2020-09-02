import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
import scipy
data =np.loadtxt("fitting.dat")
columns ={}
for column_count in range(len(data[1,:])):
    columns[column_count] = data[:,column_count]
colors = ['k', 'b', 'r', 'crimson', 'aquamarine', 'firebrick', 'orange', 'mediumorchid', 'chocolate']
time = columns[0]
N = len(time)
A0 = 1.05
B0 = -0.105
legend = []
standard = 1.05*sp.jn(2,time) - 0.105*time
sigma = np.logspace(-1,-3,9).reshape(1,-1)
for i in range(1,10):
        plt.plot(time, columns[i],colors[i-1])
        # plt.errorbar(time[::5], columns[i][::5],yerr = 0.1,fmt = 'o',capsize = 3)
        temp ='f'+f'{i}'+'(t)'
        legend.append(temp)
        plt.grid(True)
legend.append("True Value")
plt.plot(time, standard, 'gold')
plt.xlabel('Time(t)')
plt.title("Noise and the True Value")
plt.legend(legend)
plt.show()
plt.plot(time,standard,'gold')
plt.title("Errorbar for the first column")
plt.errorbar(time[::5], columns[1][::5],sigma[0][0],fmt = 'o',capsize = 3)
plt.xlabel('Time(t)')
plt.ylabel("Column 1")
plt.grid(True)
plt.legend(['True value', 'Noise'])
plt.show()

#g(t;A,B) using the definition of g
def g_direct_calculation(t,A,B):
    for m in range(len(A)):
        for n in range(len(B)):
            return (A[m]*sp.jn(2,t)+B[n]*t)

#g(t;A,B) using the matrix method
def g_using_matrix(t,A,B):
    time = np.asanyarray(t).reshape(-1,1)
    value = sp.jn(2,time)
    M=np.hstack((value,time))
    p= np.array([A,B]).reshape(2,1)
    return np.dot(M,p)

#Creating the matrix M
def create_mat_M(t):
    time = np.asanyarray(t).reshape(-1,1)
    value = sp.jn(2,time)
    M=np.hstack((value,time))
    return(M)


p = g_using_matrix(time, 1,2)
a = np.arange(0,2.01,0.1)
b = np.arange(-0.2,0.001,0.01)
b[abs(b)<1e-5]=0
len_a = len(a)
len_b = len(b)


#Calculating the error matrix
error_matrix = np.zeros((len_a,len_b))
for i in range(len_a):
        for j in range(len_b):
                error_matrix[i,j] =np.sum(((columns[1].reshape(-1,1)-g_using_matrix(time, a[i], b[j]))**2))/N


#This part of the code takes care of plotting.


i = np.arange(21)
j = np.arange(21)
levels = [0.03, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.26,0.31, 0.35]
ind = np.unravel_index(np.argmin(error_matrix, axis=None), error_matrix.shape)
temp =plt.contour(i,j,error_matrix,levels)
ind = np.asanyarray(ind)   
plt.plot(ind[0], ind[1], marker = 'o', markersize = 3,color ='red')
plt.legend(["Minima"])
plt.clabel(temp, inline=True, fontsize=6)
plt.xlabel("i")
plt.ylabel('j')
plt.title("error_matrix[i,j] for f1(t)")
plt.colorbar()
plt.show()

M = create_mat_M(time)
#Using the least squared function
A = {}
B = {}
for n in range(1,len(data[1,:])):
    [A[n], B[n]] , *rest_of_junk_produced = scipy.linalg.lstsq(M, columns[n].reshape(-1,1))     #This was there in the syntax

error_a = []
error_b = []

[error_a.append(abs(A0-A[p])) for p in range(1,len(A)+1)]
[error_b.append(abs(B0-B[p])) for p in range(1,len(B)+1)]

plt.stem((np.transpose(sigma)),np.asanyarray(error_a), 'y-o', use_line_collection=True)
# plt.yscale("log")
plt.stem((np.transpose(sigma)),np.asanyarray(error_b), 'k--', use_line_collection=True)
# plt.yscale("log")
plt.xlabel("Sigma")
plt.ylabel("Absolute Error")
plt.title("Error as a function of Standard Deviation")
plt.legend(['error_a', "error_b"])
plt.grid(True)
plt.show()
plt.stem((np.log(np.transpose(sigma))),np.asanyarray(error_a), 'y-o', use_line_collection=True)
plt.yscale("log")
plt.stem(np.log((np.transpose(sigma))),np.asanyarray(error_b), 'k--', use_line_collection=True)
plt.yscale("log")
plt.xlabel("Log(Sigma)")
plt.ylabel("Absolute Error")
plt.title("Error as a function of Standard Deviation")
plt.legend(['error_a', "error_b"])
plt.grid(True)
plt.show()