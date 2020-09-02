import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
pi = np.pi

interval_1 = np.arange(0,2*pi,0.01)
interval_2 = np.arange(-2*pi, 4*pi, 0.01)

exp = lambda x:np.exp(x)
cos_cos = lambda x:np.cos(np.cos(x))

plt.plot(interval_2, cos_cos(interval_2), 'k--')
plt.grid(True)
plt.ylabel("Cos(Cos(x))")
plt.xlabel('x')
plt.show()

plt.semilogy(interval_2, exp(interval_2),'y')
plt.grid(True)
plt.ylabel("log(exp(x))")
plt.xlabel('x')
plt.show()

a_vals_calc_exp = lambda x,k:exp(x)*np.cos(k*x)
b_vals_calc_exp = lambda x,k:exp(x)*np.sin(k*x)
a_vals_calc_cos_cos = lambda x,k:cos_cos(x)*np.cos(k*x)
b_vals_calc_cos_cos = lambda x,k:cos_cos(x)*np.sin(k*x)

N = 52

a_values_exp = np.asanyarray([(sp.integrate.quad(a_vals_calc_exp,0,2*pi, args = k))[0]/pi for k in range(N)]).reshape(-1,1)
b_values_exp = np.asanyarray([(sp.integrate.quad(b_vals_calc_exp,0,2*pi, args = k))[0]/pi for k in range(1,N)]).reshape(-1,1)
a_values_cos_cos = np.asanyarray([(sp.integrate.quad(a_vals_calc_cos_cos,0,2*pi, args = k))[0]/pi for k in range(N)]).reshape(-1,1)
b_values_cos_cos = np.asanyarray([(sp.integrate.quad(b_vals_calc_cos_cos,0,2*pi, args = k))[0]/pi for k in range(1,N)]).reshape(-1,1)

a_values_exp[0] = a_values_exp[0]/2
a_values_cos_cos[0] = a_values_cos_cos[0]/2
# a_values_cos_cos[abs(a_values_cos_cos)<1e-10] = 0
# b_values_cos_cos[abs(b_values_cos_cos)<1e-10] = 0
# a_values_exp[abs(a_values_exp)<1e-10] = 0
# b_values_exp[abs(b_values_exp)<1e-10] = 0

cos_cos_to_plot = [a_values_cos_cos[0]]
exp_to_plot = [a_values_exp[0]]

for k in range(int(N/2)-1):
    cos_cos_to_plot.append(a_values_cos_cos[k+1])
    cos_cos_to_plot.append(b_values_cos_cos[k])
    exp_to_plot.append(a_values_exp[k+1])
    exp_to_plot.append(b_values_exp[k])

cos_cos_to_plot = (np.asanyarray(cos_cos_to_plot).reshape(-1,1))
exp_to_plot = (np.asanyarray(exp_to_plot).reshape(-1,1))

plt.loglog(np.arange(N-1).reshape(-1,1), abs(exp_to_plot), 'ro')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("logarithmic plot of coefficients for exp(x)")
plt.show()

plt.semilogy(np.arange(N-1).reshape(-1,1), abs(exp_to_plot), 'ro')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("semilogarithmic plot of coefficients for exp(x)")
plt.show()

plt.loglog(np.arange(N-1).reshape(-1,1), abs(cos_cos_to_plot), 'ro')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("logarithmic plot of coefficients for cos(Cos(x))")
plt.show()

plt.semilogy(np.arange(N-1).reshape(-1,1), abs(cos_cos_to_plot), 'ro')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("semilogarithmic plot of coefficients for cos(Cos(x))")
plt.show()

x = np.linspace(0,2*pi, 401)
x = x[:-1]
rows = len(x)
columns = len(exp_to_plot)
A = np.zeros(rows*columns).reshape(rows, columns)
A[:,0] = 1
for k in range(1,int(N/2)):
    A[:, 2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)
A[abs(A)<1e-10] = 0

b_exp = exp(x)
b_cos_cos = cos_cos(x)
c_exp = np.asanyarray(sp.linalg.lstsq(A,b_exp)[0]).reshape(-1,1)   
c_cos_cos = np.asanyarray(sp.linalg.lstsq(A, b_cos_cos)[0]).reshape(-1,1)

plt.semilogy(np.arange(N-1).reshape(-1,1), abs(c_cos_cos), 'go')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("semilogarithmic plot of coefficients of Cos(Cos(x)) generated using lstsq")
plt.show()


plt.semilogy(np.arange(N-1).reshape(-1,1), abs(c_exp), 'go')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("semilogarithmic plot of coefficients of exp(x) generated using lstsq")
plt.show()

plt.loglog(np.arange(N-1).reshape(-1,1), abs(c_cos_cos), 'go')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("logarithmic plot of coefficients of Cos(Cos(x)) generated using lstsq")
plt.show()


plt.loglog(np.arange(N-1).reshape(-1,1), abs(c_exp), 'go')
plt.xlabel("n")
plt.grid(True)
plt.ylabel("Fourier Coefficients")
plt.title("logarithmic plot of coefficients of exp(x) generated using lstsq")
plt.show()

deviation_in_exp = abs(c_exp - exp_to_plot)
deviation_in_cos_cos = abs(c_cos_cos - cos_cos_to_plot)
plt.loglog(np.arange(N-1).reshape(-1,1), deviation_in_exp, 'ro')
plt.loglog(np.arange(N-1).reshape(-1,1), deviation_in_cos_cos, 'go')
plt.ylabel('Deviation')
plt.xlabel('N')
plt.grid(True)
plt.legend(['exp(x)', 'Cos(Cos(x))'])
plt.title("Deviation of the coefficients")
plt.show()

plt.semilogy(np.arange(N-1).reshape(-1,1), deviation_in_exp, 'ro')
plt.semilogy(np.arange(N-1).reshape(-1,1), deviation_in_cos_cos, 'go')
plt.ylabel('Deviation')
plt.xlabel('N')
plt.grid(True)
plt.legend(['exp(x)', 'Cos(Cos(x))'])
plt.title("semiogarithmic Deviations")
plt.show()

print("The maximum deviation in the two sets of coeffficients of cos(cos(x)) is",np.max(deviation_in_cos_cos),'.\n')
print("The maximum deviation in the two sets of coefficients of exp(x) is",np.max(deviation_in_exp),'.\n')

plt.plot(x[::10], cos_cos(x[::10]), 'k-o')
plt.stem(x[::10],np.dot(A,c_cos_cos)[::10], 'r', use_line_collection=True)
plt.grid(True)
plt.title("Predicted vs True value for cos(cos(x))")
plt.ylabel("Cos(Cos(x))")
plt.legend(['True value', 'least sq'])
plt.xlabel('x')
plt.show()

plt.semilogy(x[::10], exp(x[::10]), 'ro')
plt.semilogy(x[::10],np.dot(A,c_exp)[::10], 'go' )
plt.grid(True)
plt.title("Predicted vs True value for exp(x)")
plt.ylabel("exp(x)")
plt.legend(['True value', 'least sq'])
plt.xlabel('x')
plt.show()
