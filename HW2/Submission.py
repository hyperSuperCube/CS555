import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
CFL = 0.5
nx = 1080 # of nodes
tf = 2
lx = 8
dx = lx/(nx-1)
BCL = 1
BCR = 2
xP = np.linspace(-1,7,nx,endpoint=True)
n = np.ones((2,nx))
n[0] = -1
# Ghost node is not stored
print()
def Init(x):
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 0:
            u[i] = 1
        elif x[i]<=1 and x[i] >=0:
            u[i] = 1+x[i]
        else:
            u[i] = 2
    return u
def u2Ex(x):
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 2:
            u[i] = 1
        elif x[i]<=5 and x[i] >=2:
            u[i] = 1/3+1/3*x[i]
        else:
            u[i] = 2
    return u
def u1Ex(x):
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 1:
            u[i] = 1
        elif x[i]<=3 and x[i] >=1:
            u[i] = 0.5+1/2*x[i]
        else:
            u[i] = 2
    return u
# if sig < 0, it is a rarefication, if sig > 0 it is a shock, if sig == 0, take either
def f(u):
    return u**2/2
def jump(u,gL,gR):
    uJ = np.zeros((2,nx))
    uJ[0,1:]  = u[1:] - u[:-1]
    uJ[-1,:-1] = u[:-1] - u[1:]
    uJ[0,0] = u[0] - gL
    uJ[-1,-1] = u[-1] - gR
    return uJ*n

# The importance of G flux is that it compute the maximum/minmum f(q) at two interfaces
def Godunove(u,gL,gR):
    fs = np.zeros((2,len(u)))
    uJ = jump(u,gL,gR)
    sig = np.sign(uJ)
    sig = np.where(sig == 0, -1, sig)
    fs[0,1:] = np.abs(np.maximum(sig[0,1:] * np.maximum(f(u[:-1]), f(u[1:])), 
                                 sig[0,1:] * np.minimum(f(u[:-1]), f(u[1:]))))
    fs[0,0] = abs(max(sig[0,0] * max(f(u[0]), f(gL)), 
                                 sig[0,0] * min(f(u[0]), f(gL))))
    fs[1,:-1] = np.abs(np.maximum(sig[1,1:] * np.maximum(f(u[:-1]), f(u[1:])), 
                                  sig[1,1:] * np.minimum(f(u[:-1]), f(u[1:]))))
    fs[-1,-1] = abs(max(sig[-1,-1] * max(f(u[-1]), f(gR)), 
                                 sig[-1,-1] * min(f(u[-1]), f(gR))))
    return fs
def LaxC(u,ghostL,ghostR):
    a = np.zeros((2,nx))
    a[0,1:] = np.maximum(abs(u[:-1]),abs(u[1:])) # a-1/2
    a[-1,:-1] = np.maximum(abs(u[:-1]),abs(u[1:])) # a+1/2
    a[0,0] = max([abs(ghostL),abs(u[0])])
    a[-1,-1] = max([abs(ghostR),abs(u[-1])])
    uJ = jump(u,ghostL,ghostR)
    return a*uJ/2


centralFlx = lambda a,b: (a+b)/2
def LLC(u,ghostL,ghostR):
    C = LaxC(u,ghostL,ghostR)
    fs = np.zeros((2,nx))
    fs[0,1:] = centralFlx(f(u[1:]), f(u[:-1]))
    fs[-1,:-1] = centralFlx(f(u[1:]), f(u[:-1]))
    fs[0,0] = centralFlx(f(u[0]), f(ghostL))
    fs[-1,-1] = centralFlx(f(u[-1]), f(ghostR))
    fs += C
    return fs
# The slop on rhe two Ghost node are always zero

phi = lambda r: np.maximum(0,np.minimum(r,1))

def surfaceReconstruction(u,ghostL,ghostR,PHI = phi):
    r = np.zeros(nx)
    r[1:-1] = (u[1:-1] - u[0:-2]+1e-15)/(u[2:] - u[1:-1]+1e-15)
    r = PHI(r)
    uCons = np.zeros((2,nx))
    for i in range(0,nx):
        if i == nx-1:
            up = ghostR
        else:
            up = u[i+1]
        uCons[0,i] = u[i] - 0.5*r[i]*(up-u[i]) # Left face construction
        uCons[1,i] = u[i] + 0.5*r[i]*(up-u[i]) # Right face construction
    fs = np.zeros((2,nx))
    fs[0,1:] = centralFlx(f(uCons[0,1:]),f(uCons[1,0:-1]))
    fs[1,:-1] = centralFlx(f(uCons[1,:-1]), f(uCons[0,1:]))
    fs[0,0] = centralFlx(f(uCons[0,0]),f(ghostL))
    fs[-1,-1] = centralFlx(f(uCons[-1,-1]),f(ghostR))
    C = LaxC(u,ghostL,ghostR)
    fs += C
    return fs

def RK2(u,flux,gL,gR,dt):
    fs = flux(u,gL,gR)
    fs1 = (fs[1] - fs[0])/dx
    uh = u - dt/2*fs1
    ghostL = 2*BCL-uh[0]
    ghostR = 2*BCR-uh[-1]
    fs = flux(uh,ghostL,ghostR)
    fs2 = (fs[1] - fs[0])/dx
    return u - dt*fs2

u = Init(xP)
u2 = u2Ex(xP)
u1 = u1Ex(xP)

# WLOG we use (ui - up)*n, n is the surface norm
def Launch(u,tf,flux):
    t = 0
    while t <= tf:
        # print(t)
        dt = CFL/(np.max(u)/dx)
        ghostL = 2*BCL-u[0]
        ghostR = 2*BCR-u[-1]
        u = RK2(u,flux,ghostL,ghostR,dt)
        t+=dt   
    return u


uGod2 = Launch(u,tf,Godunove)
uLLC2 = Launch(u,tf,LLC)
uREA2 = Launch(u,tf,surfaceReconstruction)

uGod1 = Launch(u,1,Godunove)
uLLC1 = Launch(u,1,LLC)
uREA1 = Launch(u,1,surfaceReconstruction)
plt.figure(figsize=(10,10),dpi = 200)
plt.plot(xP,uGod2,label="Godunove, t = 2")
plt.plot(xP,uLLC2,label="LLC, t = 2")
plt.plot(xP,uREA2,label="Minmod Reconstruction, t = 2")
plt.plot(xP,u2,'r--',label="Exact solution, t = 2")
plt.plot(xP,uGod1,label="Godunove, t = 1",c = 'cyan')
plt.plot(xP,uLLC1,label="LLC, t = 1",c='purple')
plt.plot(xP,uREA1,label="Minmod Reconstruction, t = 1",c = 'magenta')
plt.plot(xP,u1,'k--',label="Exact solution, t = 1")
plt.legend()
plt.show()
L2G1 = la.norm((u1-uGod1))**2/nx
L2L1 = la.norm((u1-uLLC1))**2/nx
L2R1 = la.norm((u1-uREA1))**2/nx

L2G2 = la.norm((u2-uGod2))**2/nx
L2L2 = la.norm((u2-uLLC2))**2/nx
L2R2 = la.norm((u2-uREA2))**2/nx

print(L2G1,L2L1,L2R1,L2G2,L2L2,L2R2)






