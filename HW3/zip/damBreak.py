import numpy as np
import shallowpy as shp
import timeInt as solve
from matplotlib import pyplot as plt
def main():
    discr = \
    shp.discritization(1,np.array([[-1,1],[-10,10]]),[120,120])
    h,u,v = shp.Initizalizer(discr,[2,1]).getVars()
    h,u,v = solve.timeIntegration(h,u,v,discr,0.5)
    gridx = discr.gridx
#    gridy = discr.gridy
#    ax = plt.axes(projection='3d')
#    X, Y = np.meshgrid(gridx,gridy)
#    ax.contour3D(X,Y,h,50)
    plt.figure(dpi=300)
    plt.plot(gridx,u,label="u")
    plt.plot(gridx,v,label="v")
    plt.plot(gridx,h,label="h")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
