import numpy as np
from matplotlib import pyplot as plt
x0, xn, y0, yn = -1, 1, -1, 1
nx = 60
ny = 60
gridx = np.linspace(x0,xn,nx + 2)
gridy = np.linspace(y0,yn,ny + 2)
dx = abs(gridx[1]-gridx[0])
dy = abs(gridy[1]-gridy[0])
h = np.zeros((len(gridy), len(gridx)))
hu = np.zeros((len(gridy), len(gridx)))
hv = np.zeros((len(gridy), len(gridx)))
g = 1
def Init(h,x,y):
    for j in range(len(y)):
        for i in range(len(x)):
            if x[i] < 0.5 and x[i] > -0.5 and y[j] < 0.5 and y[j] > -0.5:
                h[j,i] = 2
            else:
                h[j,i] = 1
    return h

h = Init(h,gridx,gridy)

def flux_x(h,hu,hv):
    return (hu, hu**2/h + g*h**2/2, hu*hv/h)

def flux_y(h,hu,hv):
    return (hv, hu*hv/h, hv**2/h+g*h**2/2)

def maxEigx(h,hu,hv):
    u = hu/h
    return np.abs(u) + np.sqrt(g*h)

def maxEigy(h,hu,hv):
    v = hv/h
    return np.abs(v) + np.sqrt(g*h)

def timeInt(h, hu, hv, nx, ny, dx, dy):
    # Prepare the wall boundary
    h[1:-1,0] = h[1:-1,1]
    h[1:-1,-1] = h[1:-1,-2]
    h[0,1:-1] = h[1,1:-1]
    h[-1,1:-1] = h[-2,1:-1]

    hu[1:-1,0] = -hu[1:-1,1]
    hu[1:-1,-1] = -hu[1:-1,-2]
    hu[0,1:-1] = hu[1,1:-1]
    hu[-1,1:-1] = hu[-2,1:-1]

    hv[1:-1,0] = hv[1:-1,1]
    hv[1:-1,-1] = hv[1:-1,-2]
    hv[0,1:-1] = -hv[1,1:-1]
    hv[-1,1:-1] = -hv[-2,1:-1]
    eigx = maxEigx(h,hu,hv)
    eigy = maxEigy(h,hu,hv)
    dt = 0.4*np.min([dx/(np.max(eigx)), dy/(np.max(eigy))])

    h_  = h.copy()
    hu_ = hu.copy()
    hv_ = hv.copy()
    hfx, hufx, hvfx = flux_x(h_,hu_,hv_)
    hfy, hufy, hvfy = flux_y(h_,hu_,hv_)
    for y in range(1,ny+1):
        for x in range(1,nx+1):

            phiE = 0.5*(hfx[y,x] + hfx[y,x+1]) - \
            0.5*max(abs(eigx[y,x]),abs(eigx[y,x+1]))*(h_[y,x+1] - h_[y,x])
            phiW = 0.5*(hfx[y,x] + hfx[y,x-1]) - \
            0.5*max(abs(eigx[y,x-1]),abs(eigx[y,x]))*(h_[y,x] - h_[y,x-1])

            phiN = 0.5*(hfy[y,x] + hfy[y+1,x]) - \
            0.5*max(abs(eigy[y,x]), abs(eigy[y+1,x]))*(h_[y+1,x] - h_[y,x])
            phiS = 0.5*(hfy[y,x] + hfy[y-1,x]) - \
            0.5*max(abs(eigy[y-1,x]), abs(eigy[y,x]))*(h_[y,x] - h_[y-1,x])

            h[y,x] = h[y,x] - dt/dx*(phiE-phiW) - dt/dy*(phiN-phiS)

            phiE = 0.5*(hufx[y,x] + hufx[y,x+1]) - \
            0.5*max(abs(eigx[y,x]),abs(eigx[y,x+1]))*(hu_[y,x+1] - hu_[y,x])
            phiW = 0.5*(hufx[y,x] + hufx[y,x-1]) - \
            0.5*max(abs(eigx[y,x-1]),abs(eigx[y,x]))*(hu_[y,x] - hu_[y,x-1])

            phiN = 0.5*(hufy[y,x] + hufy[y+1,x]) - \
            0.5*max(abs(eigy[y,x]), abs(eigy[y+1,x]))*(hu_[y+1,x] - hu_[y,x])
            phiS = 0.5*(hufy[y,x] + hufy[y-1,x]) - \
            0.5*max(abs(eigy[y-1,x]), abs(eigy[y,x]))*(hu_[y,x] - hu_[y-1,x])

            hu[y,x] = hu[y,x] - dt/dx*(phiE-phiW) - dt/dy*(phiN-phiS)
        
            phiE = 0.5*(hvfx[y,x] + hvfx[y,x+1]) - \
            0.5*max(abs(eigx[y,x]),abs(eigx[y,x+1]))*(hv_[y,x+1] - hv_[y,x])
            phiW = 0.5*(hvfx[y,x] + hvfx[y,x-1]) - \
            0.5*max(abs(eigx[y,x-1]),abs(eigx[y,x]))*(hv_[y,x] - hv_[y,x-1])

            phiN = 0.5*(hvfy[y,x] + hvfy[y+1,x]) - \
            0.5*max(abs(eigy[y,x]), abs(eigy[y+1,x]))*(hv_[y+1,x] - hv_[y,x])
            phiS = 0.5*(hvfy[y,x] + hvfy[y-1,x]) - \
            0.5*max(abs(eigy[y-1,x]), abs(eigy[y,x]))*(hv_[y,x] - hv_[y-1,x])

            hv[y,x] = hv[y,x] - dt/dx*(phiE-phiW) - dt/dy*(phiN-phiS)
    return h, hu, hv, dt
tf = 1.35
t = 0
total = []
tc = []
while t < tf:
    print(t)
    h, hu, hv, dt = timeInt(h,hu,hv,nx,ny,dx,dy)
    da = dx*dy
    total.append(np.sum(da*h[1:-1,1:-1]))
    tc.append(t)
    t += dt
plt.figure(dpi = 300)
plt.plot(np.array(tc),np.array(total),label = "Total water mass")
plt.xlabel('time')
plt.ylabel('water mass')
plt.show()
ax = plt.axes(projection='3d')
X,Y = np.meshgrid(gridx[1:-1],gridy[1:-1])
ax.plot_surface(X,Y,h[1:-1,1:-1])
plt.show()





