import numpy as np 
import numpy.linalg as la
from matplotlib import pyplot as plt
a = 1
lx = 1
nx = 201
dt = 0.001
tf = 2
ti = 0
u = np.zeros(nx)
x = np.linspace(0,lx,nx)
dx = x[1] - x[0]
u = np.exp(-100*(x-0.5)**2)
uf = np.exp(-100*(x-0.5)**2)
P = np.zeros((nx,nx))
M = np.zeros((nx,nx))
for i in range(nx):
    for j in range(nx):
        if i == j:
            P[i,j] = 1
            M[i,j] = 1
        elif i == j+1:
            P[i,j] = -a*dt/4/dx
            M[i,j] = a*dt/4/dx
        elif j == i+1:
            P[i,j] = a*dt/4/dx
            M[i,j] = -a*dt/4/dx
P[0,-1] = -a*dt/4/dx
P[-1,0] = a*dt/4/dx
M[0,-1] = a*dt/4/dx
M[-1,0] = -a*dt/4/dx
while ti < tf:
    u = la.solve(P,M@u)
    ti += dt
plt.plot(x,u,label='Exact Solution')
plt.plot(x,uf,label='Numerical Solution')
plt.legend()
plt.show()

