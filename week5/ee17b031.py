#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from scipy.signal import convolve2d
import time
from scipy.linalg import lstsq
import sys
from matplotlib import cm
arguments = sys.argv
 

#Variable Declaration
if len(arguments)==1:
    Nx = 50
    Ny = 50                     #In case of no assignment, it will default to these values
    radius = 0.35
    Niter = 1500
elif len(arguments) == 5:
    Nx = int(arguments[1])
    Ny = int(arguments[2])
    radius = float(arguments[3])
    Niter = int(arguments[4])
else:
    print("Invalid number of command line arguments.")
    sys.exit(0)

#plotting potential values for the electrode
x = np.arange(Nx+1)
y = np.arange(Ny+1)
X,Y = np.meshgrid(x,y)
Y = Y[::-1]
X = (X-Nx/2)/Nx
Y = (Y-Ny/2)/Ny
potential  = np.zeros((Nx+1, Ny+1))
potential[X*X+Y*Y<=radius*radius]= 1.0
electrode = potential.copy()
plate = 1-electrode
plt.scatter(X,Y,electrode,'r', linewidths= 3)
plt.scatter(X,Y,plate, 'b', linewidths=0.5)
plt.title("Potential after initialization of the electrode")
plt.legend(['electrode','metal plate'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Trying Something different
AVERAGE_FILTER = np.array([[0,0.25,0],[0.25,0,0.25],[0,0.25,0]])

error = np.zeros(Niter)
#iteration
old_potential = potential.copy()
for i in range(Niter):
    old_potential = potential.copy()
    potential[1:-1, 1 :-1] = 0.25*(potential[1:-1, 2:]+potential[1:-1, 0:-2]+potential[2:, 1:-1]+potential[0:-2, 1:-1])
    # potential= convolve2d(potential, AVERAGE_FILTER, mode='same')             #using the filter
    potential[1:-1, 0] = potential[1:-1,1]
    potential[1:-1, -1] = potential[1:-1,-2]
    potential[0,1:-1] = potential[1,1:-1]
    potential[X*X+Y*Y<=radius*radius]= 1.0
    error[i] = np.max(abs(potential-old_potential))

#making the fit1 and fit2 functions
log_errors = np.log(error).reshape(-1,1)
number_of_iterations = np.arange(1,Niter+1).reshape(-1,1)
ones = np.ones(Niter).reshape(-1,1)
fitting_matrix = np.hstack((ones, number_of_iterations))
fit1_coefficients, *rest_of_junk_produced= lstsq(fitting_matrix, log_errors)
fit2_coefficients, *rest_of_junk_produced= lstsq(fitting_matrix[500:], log_errors[500:])

#loglog plot
plt.loglog(np.arange(1,Niter+1), error,'k')
plt.xlabel('$nth$ iteration')
plt.title("Logarithmic plot of errors for all iterations")
plt.ylabel('Error')
plt.legend(['errors'])
plt.grid(True)
plt.show()

#fitting and plotting along with the initial curve
fit1 = np.dot(fitting_matrix,fit1_coefficients)
fit2 = np.dot(fitting_matrix[500:],fit2_coefficients)

plt.semilogy(np.arange(1,Niter+1), error,'k')
plt.semilogy(number_of_iterations,np.exp(fit1), 'y', linewidth = 7, alpha = 0.7)
plt.xlabel('$nth$ iteration')
plt.title("Semilogarithmic plot of errors for all iterations")
plt.ylabel('Error')
plt.grid(True)
plt.legend(['errors','fit1'])
plt.show()

plt.semilogy(np.arange(1,Niter+1)[500:], error[500:],'k')
plt.semilogy(number_of_iterations[500:],np.exp(fit2), 'y', linewidth = 7, alpha = 0.7)
plt.xlabel('$nth$ iteration')
plt.title("Semilogarithmic plot of errors for iterations after 500")
plt.ylabel('Error')
plt.grid(True)
plt.legend(['errors','fit2'])
plt.show()

#3d plotting
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf_plot = ax.plot_surface(X,Y,potential,rstride = 1, cstride = 1,alpha = 0.7, cmap=cm.coolwarm)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title("Potential as a function of x and y")
fig.colorbar(surf_plot, shrink=0.5, aspect=5)
# ax.set_zlabel('potential', fontsize = 20, rotation = 60)
plt.show()


#contour_plot_potential
temp = plt.contour(X,Y,potential)
plt.title("Contour plot of the potential")
plt.ylabel("y")
plt.xlabel('x')
plt.clabel(temp, inline = True, fontsize =6)
plt.scatter(X,Y,electrode,'r', linewidths= 3)
plt.scatter(X,Y,plate, 'gold', linewidths=0.3)
plt.legend(['electrode', 'metal plate'])
plt.show()


#plotting the current
Jx = np.zeros_like(potential)
Jy = np.zeros_like(potential)
Jx[:,1:-1] = 0.5*(potential[:,0:-2]-potential[:,2:])
Jy[1:-1] = 0.5*(potential[2:, :]-potential[0:-2,:])
plt.quiver(X,Y,Jx,Jy,scale = 4e0)
plt.title("Current Density")
plt.xlabel("x")
plt.ylabel('y')
plt.scatter(X,Y,electrode,'r', linewidths= 3)
plt.scatter(X,Y,plate, 'gold', linewidths=0.3)
plt.legend(["current direction",'electrode','metal plate'])
plt.show()

