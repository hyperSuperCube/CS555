import numpy as np
from matplotlib import pyplot as plt
x0, xn = -1, 1
nx = 60
gridx = np.linspace(x0,xn,nx + 2)
dx = abs(gridx[1]-gridx[0])
h = np.zeros(len(gridx))
hu = np.zeros(len(gridx))
hv = np.zeros(len(gridx))
g = 1
def Init(h,x):
    for i in range(len(x)):
        if x[i] < 0.5 and x[i] > -0.5:
            h[i] = 2
        else:
            h[i] = 1
    return h

h = Init(h,gridx)

def flux_x(h,hu,hv):
    return (hu, hu**2/h + g*h**2/2, hu*hv/h)

def maxEigx(h,hu,hv):
    u = hu/h
    return np.abs(u) + np.sqrt(g*h)

def timeInt(h, hu, hv, nx, dx):
    # Prepare the wall boundary
    h[0] = h[1]
    h[-1] = h[-2]

    hu[0] = -hu[1]
    hu[-1] = -hu[-2]

    hv[0] = hv[1]
    hv[-1] = hv[-2]

    eigx = maxEigx(h,hu,hv)
    dt = 0.4*dx/(np.max(eigx))

    h_  = h.copy()
    hu_ = hu.copy()
    hv_ = hv.copy()
    hfx, hufx, hvfx = flux_x(h_,hu_,hv_)
    for x in range(1,nx+1):
        phiE = 0.5*(hfx[x] + hfx[x+1]) - \
        0.5*max(abs(eigx[x]),abs(eigx[x+1]))*(h_[x+1] - h_[x])
        phiW = 0.5*(hfx[x] + hfx[x-1]) - \
        0.5*max(abs(eigx[x-1]),abs(eigx[x]))*(h_[x] - h_[x-1])

        h[x] = h[x] - dt/dx*(phiE-phiW)

        phiE = 0.5*(hufx[x] + hufx[x+1]) - \
        0.5*max(abs(eigx[x]),abs(eigx[x+1]))*(hu_[x+1] - hu_[x])
        phiW = 0.5*(hufx[x] + hufx[x-1]) - \
        0.5*max(abs(eigx[x-1]),abs(eigx[x]))*(hu_[x] - hu_[x-1])

        hu[x] = hu[x] - dt/dx*(phiE-phiW)
    
        phiE = 0.5*(hvfx[x] + hvfx[x+1]) - \
        0.5*max(abs(eigx[x]),abs(eigx[x+1]))*(hv_[x+1] - hv_[x])
        phiW = 0.5*(hvfx[x] + hvfx[x-1]) - \
        0.5*max(abs(eigx[x-1]),abs(eigx[x]))*(hv_[x] - hv_[x-1])

        hv[x] = hv[x] - dt/dx*(phiE-phiW)
    return h, hu, hv, dt
tf = 1.35
t = 0

while t < tf:
    print(t)
    h, hu, hv, dt = timeInt(h,hu,hv,nx,dx)
    t += dt
u = hu/h
v = hv/h
plt.figure(dpi=300)
plt.plot(gridx,u,label="u")
plt.plot(gridx,v,label="v")
plt.plot(gridx,h,label="h")
plt.legend()
plt.show()





