import numpy as np
import shallowpy as shp
rk4a = np.array([ 0.0 ,
-567301805773.0/1357537059087.0 ,
-2404267990393.0/2016746695238.0 ,
-3550918686646.0/2091501179385.0 ,
-1275806237668.0/842570457699.0])

rk4b = [ 1432997174477.0/9575080441755.0,
5161836677717.0/13612068292357.0,
1720146321549.0/2090206949498.0,
3134564353537.0/4481467310338.0,
2277821191437.0/14882151754819.0]

rk4c = [ 0.0,
1432997174477.0/9575080441755.0,
2526269341429.0/6820363962896.0,
2006345519317.0/3224310063776.0,
2802321613138.0/2924317926251.0]

def timeIntegration(dv1,dv2,dv3,discr,tf):
    t = 0 
    res1,res2,res3 = \
    np.zeros(dv1.shape),np.zeros(dv1.shape),np.zeros(dv1.shape)
    while t < tf:
        print(t)
        q1,q2,q3 = shp.makeFulidState(dv1,dv2,dv3,discr).q
        for iter in range(5):
            rhsq1, rhsq2, rhsq3, dt = \
            shp.makeFulidState(dv1,dv2,dv3,discr).my_rhs()
            res1 = rk4a[iter]*res1 + dt*rhsq1
            q1 += rk4b[iter]*res1

            res2 = rk4a[iter]*res2 + dt*rhsq2
            q2 += rk4b[iter]*res2

            res3 = rk4a[iter]*res3 + dt*rhsq3
            q3 += rk4b[iter]*res3
            dv1,dv2,dv3 = shp.cvTodv(q1,q2,q3)
        t += dt
    return dv1,dv2,dv3



