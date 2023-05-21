import numpy as np
from matplotlib import pyplot as plt
class discritization():
    def __init__(self, dim, domain, nele):
        """
        Parameter:: 
            dim::  The dimension of the problem
            domain:: \
            The size of the region that is going to be discretized ((x0, xn),
                                                                    (y0, yn))
            nele:: # of elements on axis x and (y)
         """
        self.dim = dim
        self.domain = domain
        if dim == 1:
            self.gridx = np.linspace(domain[0,0],domain[0,1],nele[0]+1)
            self.dx = abs(self.gridx[1]-self.gridx[0])
            self.u = np.zeros(self.gridx.shape)
            self.v = np.zeros(self.gridx.shape)
            self.h = np.zeros(self.gridx.shape)
        if dim == 2:
            self.gridx = np.linspace(domain[0][0],domain[0][1],nele[0]+1)
            self.gridy = np.linspace(domain[1][0],domain[1][1],nele[1]+1)
            self.dx = abs(self.gridx[1]-self.gridx[0])
            self.dy = abs(self.gridy[1]-self.gridy[0])
            self.u = np.zeros((nele[0]+1,nele[1]+1))
            self.v = np.zeros((nele[0]+1,nele[1]+1))
            self.h = np.zeros((nele[0]+1,nele[1]+1))
    def getVars(self):
        return self.h, self.u, self.v

class Initizalizer():
    def __init__(self,discr,Ih):
        self.discr = discr
        self.u = discr.u
        self.v = discr.v
        self.h = discr.h
        if self.discr.dim == 1:
            self.gridx = self.discr.gridx
            for i in range(len(self.gridx)):
                if self.gridx[i] < 0:
                    self.h[i] = Ih[0]
                else:
                    self.h[i] = Ih[1]
        elif self.discr.dim == 2:
            self.gridx = self.discr.gridx
            self.gridy = self.discr.gridy
            for col in range(len(self.gridx)):
                for row in range(len(self.gridy)):
                    if self.gridx[col] < 0.5\
                     and self.gridx[col] > -0.5\
                          and self.gridy[row] > -0.5\
                           and self.gridy[row] < 0.5:
                        self.h[row,col] = Ih[0]
                    else:
                        self.h[row,col] = Ih[1]
    def getVars(self):
        return self.h, self.u, self.v

class makeFulidState():
    def __init__(self, h,u,v, discr):
        self.u = u
        self.discr = discr
        self.v = v
        self.h = h
        self.q1,self.q2,self.q3 = dvTocv(self.h,self.u,self.v)
        self.q = (self.q1,self.q2,self.q3)
        self.dim = discr.dim
        if self.dim == 1:
            self.Kx = len(u)
        elif self.dim == 2:
            def wallBoundary():
                h,u,v = cvTodv(self.q1,self.q2,self.q3)
                L_bc_h = h[:,0]
                R_bc_h = h[:,-1]
                U_bc_h = h[0,:]
                B_bc_h = h[-1,:]

                L_bc_u = -u[:,0]
                R_bc_u = -u[:,-1]
                U_bc_u = u[0,:]
                B_bc_u = u[-1,:]

                L_bc_v = v[:,0]
                R_bc_v = v[:,-1]
                U_bc_v = -v[0,:]
                B_bc_v = -v[-1,:]

                L_bc_q1, L_bc_q2, L_bc_q3 = \
                dvTocv(L_bc_h, L_bc_u, L_bc_v)
                R_bc_q1, R_bc_q2, R_bc_q3 = \
                dvTocv(R_bc_h, R_bc_u, R_bc_v)
                U_bc_q1, U_bc_q2, U_bc_q3 = \
                dvTocv(U_bc_h, U_bc_u, U_bc_v)
                B_bc_q1, B_bc_q2, B_bc_q3 = \
                dvTocv(B_bc_h, B_bc_u, B_bc_v)

                return (L_bc_q1, R_bc_q1, U_bc_q1, B_bc_q1), \
                (L_bc_q2, R_bc_q2, U_bc_q2, B_bc_q2), \
                (L_bc_q3, R_bc_q3, U_bc_q3, B_bc_q3)

            self.Kx = u.shape[1]
            self.Ky = u.shape[0]
            self.q1bc, self.q2bc, self.q3bc = wallBoundary()
    def maxEigSpeed(self):
        if self.dim == 1:
            eigx = np.abs(self.u) + np.sqrt(1*self.h)
            maxEig = np.zeros((2,len(self.u)))
            maxEig[0,1:] = \
            np.maximum(np.abs(eigx[:-1]),np.abs(eigx[1:]))
            maxEig[-1,:-1] = \
            np.maximum(np.abs(eigx[:-1]),np.abs(eigx[1:]))
            return maxEig,0
        elif self.dim == 2:
            Kx = self.u.shape[1]
            Ky = self.u.shape[0]
            eigx = np.abs(self.u) + np.sqrt(1*self.h)
            eigy = np.abs(self.v) + np.sqrt(1*self.h)

            maxEigx = np.zeros((2,Kx,Ky)) 
            # np.ndarray is used to store each trace pair on a specific axis
            #, here we have Ky sets of Kx pairs 
            maxEigy = np.zeros((2,Ky,Kx))
            for ky in range(Ky):
                maxEigx[0,1:,ky] = \
                np.maximum(np.abs(eigx[ky,:-1]),np.abs(eigx[ky,1:]))
                maxEigx[-1,:-1,ky] = \
                np.maximum(np.abs(eigx[ky,:-1]),np.abs(eigx[ky,1:]))
                
                maxEigx[0,0,ky] = eigx[ky,0]
                maxEigx[-1,-1,ky] = eigx[ky,-1]

            for kx in range(Kx):
                maxEigy[0,1:,kx] = \
                np.maximum(np.abs(eigy[:-1,kx]),np.abs(eigy[1:,kx]))
                maxEigy[-1,:-1,kx] = \
                np.maximum(np.abs(eigy[:-1,kx]),np.abs(eigy[1:,kx]))

                maxEigy[0,0,kx] = eigy[0,kx]
                maxEigy[-1,-1,kx] = eigy[-1,kx]

            return maxEigx,maxEigy
        
    def autoTiming(self):
        if self.dim == 1:
            dx = self.discr.dx
            maxEig, a= self.maxEigSpeed()
            dt = 0.5*np.min(dx/maxEig)
        elif self.dim == 2:
            dx = self.discr.dx
            dy = self.discr.dy
            maxEigx, maxEigy = self.maxEigSpeed()
            dt = 0.4*np.min([dx/(np.max(maxEigx)), dy/(np.max(maxEigy))])
        return dt
    

    def flux(self):
        if self.dim == 1:
            f1x = self.q2
            f2x = self.q2**2/self.q1 + 1*self.q1**2/2
            f3x = self.q2*self.q3/self.q1
            return f1x,f2x,f3x
        elif self.dim == 2:
            f1x = self.q2
            f2x = self.q2**2/self.q1 + 1*self.q1**2/2
            f3x = self.q2*self.q3/self.q1

            f1y = self.q3
            f2y = self.q2*self.q3/self.q1
            f3y = self.q3**2/self.q1 + 1*self.q1**2/2
            return f1x,f2x,f3x,f1y,f2y,f3y


    def jump(self):
        if self.dim == 1:
            nx = np.ones((2,len(self.u)))
            nx[0] = -1
            q1J, q2J, q3J = broadCasting(self.q1, self.q2, self.q3,Jump)*nx
            return q1J, q2J, q3J
        elif self.dim == 2:
            Kx = self.u.shape[1]
            Ky = self.u.shape[0]
            nx = np.ones((2,Kx))
            nx[0] = -1
            ny = np.ones((2,Ky))
            nx[0] = -1
            q1xJ, q2xJ, q3xJ = \
            np.zeros((2,Kx,Ky)), np.zeros((2,Kx,Ky)), np.zeros((2,Kx,Ky))
            q1yJ, q2yJ, q3yJ = \
            np.zeros((2,Ky,Kx)), np.zeros((2,Ky,Kx)), np.zeros((2,Ky,Kx))
            for ky in range(Ky):
                q1xJ[:,:,ky],  q2xJ[:,:,ky], q3xJ[:,:,ky] = \
                    broadCasting(self.q1[ky,:], self.q2[ky,:], self.q3[ky,:],Jump)
                """
                note:: 
                    qbc[0]:: Left
                    qbc[1]:: Right
                    qbc[2]:: Up
                    qbc[3]:: Buttom

                """
                q1xJ[0,0,ky] = self.q1[ky,0]-self.q1bc[0][ky]
                q1xJ[-1,-1,ky] = self.q1[ky,-1]-self.q1bc[1][ky]

                q2xJ[0,0,ky] = self.q2[ky,0]-self.q2bc[0][ky]
                q2xJ[-1,-1,ky] = self.q2[ky,-1]-self.q2bc[1][ky]

                q3xJ[0,0,ky] = self.q3[ky,0]-self.q3bc[0][ky]
                q3xJ[-1,-1,ky] = self.q3[ky,-1]-self.q3bc[1][ky]

                q1xJ[:,:,ky],  q2xJ[:,:,ky], q3xJ[:,:,ky] = \
                nx*(q1xJ[:,:,ky],  q2xJ[:,:,ky], q3xJ[:,:,ky])

            for kx in range(Kx):
                q1yJ[:,:,kx], q2yJ[:,:,kx], q3yJ[:,:,kx] = \
                    broadCasting(self.q1[:,kx], self.q2[:,kx], self.q3[:,kx],Jump)
                
                q1yJ[0,0,kx] = self.q1[0,kx]-self.q1bc[2][kx]
                q1yJ[-1,-1,kx] = self.q1[-1,kx]-self.q1bc[3][kx]

                q2yJ[0,0,kx] = self.q2[0,kx]-self.q2bc[2][kx]
                q2yJ[-1,-1,kx] = self.q2[-1,kx]-self.q2bc[3][kx]

                q3yJ[0,0,kx] = self.q3[0,kx]-self.q3bc[2][kx]
                q3yJ[-1,-1,kx] = self.q3[-1,kx]-self.q3bc[3][kx]

                q1yJ[:,:,kx],  q2yJ[:,:,kx], q3yJ[:,:,kx] = \
                ny*(q1yJ[:,:,kx],  q2yJ[:,:,kx], q3yJ[:,:,kx])
            return q1xJ, q2xJ, q3xJ, q1yJ, q2yJ, q3yJ
        

    def interfacial_flux(self):
        if self.dim == 1:
            f1x,f2x,f3x = self.flux()
            q1f,q2f,q3f = broadCasting(f1x,f2x,f3x,centralFlx)
            maxEigx,a = self.maxEigSpeed()
            q1J,q2J,q3J = self.jump()
            q1f += (maxEigx*q1J/2)
            q2f += (maxEigx*q2J/2)
            q3f += (maxEigx*q3J/2)
            return q1f,q2f,q3f
        elif self.dim == 2:
            Kx = self.u.shape[1]
            Ky = self.u.shape[0]
            f1x,f2x,f3x,f1y,f2y,f3y = self.flux()
            q1xf, q2xf, q3xf = \
            np.zeros((2,Kx,Ky)),np.zeros((2,Kx,Ky)),np.zeros((2,Kx,Ky))
            q1yf, q2yf, q3yf = \
            np.zeros((2,Ky,Kx)),np.zeros((2,Ky,Kx)),np.zeros((2,Ky,Kx))
            for ky in range(Ky):
                q1xf[:,:,ky], q2xf[:,:,ky], q3xf[:,:,ky] = \
                    broadCasting(f1x[ky,:], f2x[ky,:], f3x[ky,:], centralFlx)
                
            for kx in range(Kx):

                q1yf[:,:,kx], q2yf[:,:,kx], q3yf[:,:,kx] = \
                    broadCasting(f1y[:,kx],f2y[:,kx],f3y[:,kx],centralFlx)
                
            maxEigx, maxEigy = self.maxEigSpeed()
            q1xJ, q2xJ, q3xJ, q1yJ, q2yJ, q3yJ = self.jump()
            q1xf += (maxEigx*q1xJ/2)
            q2xf += (maxEigx*q2xJ/2)
            q3xf += (maxEigx*q3xJ/2)
            q1yf += (maxEigy*q1yJ/2)
            q2yf += (maxEigy*q2yJ/2)
            q3yf += (maxEigy*q3yJ/2)
            return q1xf, q2xf, q3xf, q1yf, q2yf, q3yf 

    def my_rhs(self):
        if self.dim == 1:
            dx = self.discr.dx
            q1f,q2f,q3f = self.interfacial_flux()
            dt = self.autoTiming()
            rhs1 = (q1f[0] - q1f[1])/dx
            rhs2 = (q2f[0] - q2f[1])/dx
            rhs3 = (q3f[0] - q3f[1])/dx
            return rhs1, rhs2, rhs3, dt
        elif self.dim == 2:
            dx = self.discr.dx
            dy = self.discr.dy

            q1xf, q2xf, q3xf, q1yf, q2yf, q3yf = self.interfacial_flux()
            dt = self.autoTiming()
            q1xf_ = (q1xf[0,:,:] - q1xf[1,:,:])/dx
            q2xf_ = (q2xf[0,:,:] - q2xf[1,:,:])/dx
            q3xf_ = (q3xf[0,:,:] - q3xf[1,:,:])/dx

            q1yf_ = (q1yf[0,:,:] - q1yf[1,:,:])/dy
            q2yf_ = (q2yf[0,:,:] - q2yf[1,:,:])/dy
            q3yf_ = (q3yf[0,:,:] - q3yf[1,:,:])/dy
            rhs1 = q1xf_ + q1yf_
            rhs2 = q2xf_ + q2yf_
            rhs3 = q3xf_ + q3yf_
            return rhs1, rhs2, rhs3, dt
    
def cvTodv(h,hu,hv):
    return h, hu/h, hv/h
def dvTocv(h,u,v):
    return h, h*u, h*v
"""
    note:: WLOG we use q.int - q.ext to specify a jump
"""
def Jump(q):
    qJ = np.zeros((2,len(q)))
    qJ[0,1:]   = q[1:] - q[:-1]
    qJ[-1,:-1] = q[:-1] - q[1:]
    return qJ

def broadCasting(q1,q2,q3,f):
    return f(q1), f(q2), f(q3)

def centralFlx(f1):
    q1 = np.zeros((2,len(f1)))
    q1[0,1:], q1[1,:-1] = (f1[:-1] + f1[1:])/2, (f1[:-1] + f1[1:])/2
    q1[0,0], q1[1,-1] = f1[0], f1[-1]
    return q1





        

