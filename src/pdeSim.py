# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:00:23 2018

@author: andrew.ferguson
"""
import numpy as np
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import coo_matrix

class driftDiff2Dvideo:
    '''
    This class solves the drift diffusion equation in 2 dimensions.
    
    '''
    def __init__(self,n,f,g,dx,dy,dt,mu,D,zero_flux=True,iterations=10,append=1):
        self.n=n
        self.f=f
        self.g=g
        self.dx=dx
        self.dy=dy
        self.dt=dt
        self.mu=mu
        self.D=D
        self.zero_flux=zero_flux
        self.iterations = iterations
        self.append = append        
        
        self.nx = self.n.shape[0]
        self.ny = self.n.shape[1]
        
        self.steps = self.nx*self.ny
                
        #These are independent of time
        self.p = self.dt*self.D/self.dx**2
        self.q = self.dt*self.D/self.dy**2
        self.r = self.dt*self.mu/self.dx/2
        self.s = self.dt*self.mu/self.dy/2
        
        print('nx=',self.nx)
        print('ny=',self.ny)        
        print('p=',self.p,'q=',self.q,'r=',self.r,'s=',self.s)
        print('iterations=',self.iterations)
        
    def buildMatrix(self):

        Identity = np.identity(self.steps)
     
        Arow = []
        Acol = []
        Adata = []
        
        for i in range(self.nx):
            for j in range(self.ny):
                Adiag = j*self.nx+i
                
                if i ==0:
                    if j==0:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(self.f[i+1,j]+self.f[i,j])+2*self.s*(self.g[i,j+1]+self.g[i,j]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+self.nx)
                        Adata.append(-2*self.q)
                        
                    elif j==self.ny-1: 
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(self.f[i+1,j]+self.f[i,j])+2*self.s*(-self.g[i,j]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag-self.nx)
                        Adata.append(-2*self.q)
                        
                    else:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(self.f[i+1,j]+self.f[i,j])+self.s*(self.g[i,j+1]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-self.nx)
                        Adata.append(-self.q-self.s*self.g[i,j])
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+self.nx)
                        Adata.append(-self.q+self.s*self.g[i,j])
                    
                elif i == self.nx-1:
                    if j==0:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(-self.f[i,j]-self.f[i-1,j])+2*self.s*(self.g[i,j+1]-self.g[i,j]))                        
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+self.nx)
                        Adata.append(-2*self.q)
                        
                        
                    elif j==self.ny-1:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(-self.f[i,j]-self.f[i-1,j])+2*self.s*(self.g[i,j]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag-self.nx)
                        Adata.append(-2*self.q)
                        
                    else:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*self.p+2*self.q+2*self.r*(-self.f[i,j]-self.f[i-1,j])+self.s*(self.g[i,j+1]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-self.nx)
                        Adata.append(-self.q-self.s*self.g[i,j])
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*self.p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+self.nx)
                        Adata.append(-self.q+self.s*self.g[i,j])
                                            
                elif j == 0:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*self.p+2*self.q+2*self.r*(self.f[i+1,j]-self.f[i-1,j])+2*self.s*(self.g[i,j+1]+self.g[i,j]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-self.p-self.r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-self.p+self.r*self.f[i,j])
                      
                    Arow.append(Adiag)
                    Acol.append(Adiag+self.nx)
                    Adata.append(-2*self.q)
                    
                elif j == self.ny-1:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*self.p+2*self.q+self.r*(self.f[i+1,j]-self.f[i-1,j])+2*self.s*(-self.g[i,j]-self.g[i,j-1]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-self.p-self.r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-self.p+self.r*self.f[i,j])
                      
                    Arow.append(Adiag)
                    Acol.append(Adiag-self.nx)
                    Adata.append(-2*self.q)
                    
                else:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*self.p+2*self.q+self.r*(self.f[i+1,j]-self.f[i-1,j])+self.s*(self.g[i,j+1]-self.g[i,j-1]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-self.p+self.r*self.f[i,j])
        
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-self.p-self.r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+self.nx)
                    Adata.append(-self.q+self.s*self.g[i,j])
        
                    Arow.append(Adiag)
                    Acol.append(Adiag-self.nx)
                    Adata.append(-self.q-self.s*self.g[i,j])
                    
        
        self.Adata_coo=coo_matrix((Adata, (Arow, Acol)), shape=(self.steps, self.steps))
        self.Adata_csr = self.Adata_coo.tocsr()
        self.Adata_dense = self.Adata_coo.todense()
        self.Bdata_dense = 4*Identity-self.Adata_dense 
        
        print('Matrix built')
        
    def cnSolve(self):
            
        n_flat = np.zeros((self.steps))    
            
        for j in range(self.ny):
            for i in range(self.nx):
                n_flat[self.nx*j+i] = self.n[i,j]
        
        output_list = []        
              
        iteration_counter = 0
        
        for p in range(self.iterations):
            if iteration_counter%self.append==0:
                output_array = np.zeros((self.nx,self.ny))  
                for j in range(self.ny):
                    for i in range(self.nx):
                        output_array[i,j]=n_flat[j*self.nx+i]
                
                output_list.append(output_array)
            
            iteration_counter +=1    
            bb = self.Bdata_dense.dot(n_flat)              
            n_flat[:] = spsolve(self.Adata_csr,bb.T) 
            
        return output_list    

class Poisson2DCN:
        '''Solves the 2D heat diffusion equation using CN method.
        '''
        
        def __init__(self,k,c,rho,q,T0,dt,dx,dy,iterations,append):
            self.k = k
            self.c = c
            self.rho = rho
            self.q = q  # The heat source density
            self.T0 = T0
            self.dx = dx
            self.dy = dy
            self.dt = dt
            
            self.nx = self.k.shape[0]
            self.ny = self.k.shape[1]
            #self.T = len(self.q)
            
            self.steps = self.nx*self.ny
            self.iterations = iterations
            self.append=append
        
        def buildMatrix(self):
                 
            Arow = []
            Acol = []
            Adata = []
            
            Brow = []
            Bcol = []
            Bdata = []
                             
            for i in range(self.nx):
                for j in range(self.ny):
                    diag = j*self.nx+i
                    
                    if i==0:
                        Arow.append(diag)
                        Acol.append(diag)
                        Adata.append(1) 
                        
                        Brow.append(diag)
                        Bcol.append(diag)
                        Bdata.append(1)  
                    elif j==0:
                        Arow.append(diag)
                        Acol.append(diag)
                        Adata.append(1)

                        Brow.append(diag)
                        Bcol.append(diag)
                        Bdata.append(1)
                          
                    elif i==self.nx-1:
                        Arow.append(diag)
                        Acol.append(diag)
                        Adata.append(1) 
                        
                        Brow.append(diag)
                        Bcol.append(diag)
                        Bdata.append(1)                        
                        
                    elif j==self.ny-1:
                        Arow.append(diag)
                        Acol.append(diag)
                        Adata.append(1)  
                        
                        Brow.append(diag)
                        Bcol.append(diag)
                        Bdata.append(1)                        
                        
                    else:
                        a = 1/self.dx**2*0.5*(self.k[i,j+1]+self.k[i,j])
                        b = 1/self.dx**2*0.5*(self.k[i+1,j+1]+self.k[i+1,j])
                        c = 1/self.dy**2*0.5*(self.k[i,j+1]+self.k[i+1,j+1])
                        d = 1/self.dy**2*0.5*(self.k[i,j]+self.k[i+1,j])
                        e = 0.25*(self.c[i,j+1]*self.rho[i,j+1]+self.c[i+1,j+1]*self.rho[i+1,j+1]+
                        self.c[i,j]*self.rho[i,j]+self.c[i+1,j]*self.rho[i+1,j])/self.dt
                        
                        
                        Arow.append(diag)
                        Acol.append(diag)
                        Adata.append(-a-b-c-d-2*e)
                        
                        Arow.append(diag)
                        Acol.append(diag+1)
                        Adata.append(b)
            
                        Arow.append(diag)
                        Acol.append(diag-1)
                        Adata.append(a)
                        
                        Arow.append(diag)
                        Acol.append(diag+self.nx)
                        Adata.append(c)
            
                        Arow.append(diag)
                        Acol.append(diag-self.nx)
                        Adata.append(d)           
                        
                        Brow.append(diag)
                        Bcol.append(diag)
                        Bdata.append(a+b+c+d-2*e)
                        
                        Brow.append(diag)
                        Bcol.append(diag+1)
                        Bdata.append(-b)
            
                        Brow.append(diag)
                        Bcol.append(diag-1)
                        Bdata.append(-a)
                        
                        Brow.append(diag)
                        Bcol.append(diag+self.nx)
                        Bdata.append(-c)
            
                        Brow.append(diag)
                        Bcol.append(diag-self.nx)
                        Bdata.append(-d)                        
                                         
            self.Adata_coo=coo_matrix((Adata, (Arow, Acol)), shape=(self.steps, self.steps))
            self.Adata_csr = self.Adata_coo.tocsr()
            self.Adata_dense = self.Adata_coo.todense()
                        
            self.Bdata_coo=coo_matrix((Bdata, (Brow, Bcol)), shape=(self.steps, self.steps))
            self.Bdata_csr = self.Bdata_coo.tocsr()
            self.Bdata_dense = self.Bdata_coo.todense()
            
            #print(self.Adata_dense)
            #print(self.Bdata_dense)
        
        def cnSolve(self):
            T0_flat = np.zeros((self.steps)) 
            q_flat = np.zeros((self.steps))

            for j in range(self.ny):
                for i in range(self.nx):
                    T0_flat[self.nx*j+i] = self.T0[i,j]
                    q_flat[self.nx*j+i] = self.q[i,j]

            u_list = [T0_flat]

            for p in range(self.iterations - 1):
                bb = self.Bdata_dense.dot(u_list[-1]) - 2 * q_flat
                u_list.append(spsolve(self.Adata_csr, bb.T))

            # Wrap up the results
            us_wrap = []
            for u in u_list:
                u_temp = np.zeros((self.nx, self.ny))
                for j in range(self.ny):
                    for i in range(self.nx):
                        u_temp[i, j] = u[self.nx * j + i]
                us_wrap.append(u_temp)

            return us_wrap

class Poisson1D:
    ''' This class solves a 1D heat diffusion equation using the Crank-Nicolson method. 
        The possibility to have a time dependent heat source is included.    
        T = 0 boundary conditions are implemented
    '''
    def __init__(self,k,c,rho,q,T0,dt,dx):
        '''
        k,c,rho are numpy array length N, specifying the obvious thermal parameters
        T0 is a numpy array length N, specifying the initial temperature
        q is a numpy array with dimensions N x T, specifying the time-dependent power dissipated per unit length
        dt is a scalar specifying the time step
        dx is a scalar specifying the distance
        '''
        self.k = k
        self.c = c
        self.rho = rho
        self.q = q
        self.T0=T0
        self.dx = dx
        self.dt = dt
        
    def CNsolve(self):
        '''
        Performs a solution via the CN method
        '''
        N = self.k.shape[0]
        T = self.q.shape[1]
            
        #Initialise the matrices
        A = np.zeros((N,N))
        B = np.zeros((N,N))
        b = np.zeros((N,T))
        Temperatures = np.zeros((N,T))
        for i in range(N):
            Temperatures[i,0] = self.T0[i]   
        
        for i in range(N):
            b[i,:]=self.q[i,:]*self.dt/self.c[i]/self.rho[i]

        for i in range(N):
            if i ==0:
                A[i,i]=1
                B[i,i]=1
            elif i ==N-1:
                A[i,i]=1
                B[i,i]=1
            else:
                A[i,i] = 2 + self.dt/self.dx**2/self.c[i]/self.rho[i]*(self.k[i]+self.k[i+1])
                A[i,i-1] =-self.dt/self.dx**2/self.c[i]/self.rho[i]*self.k[i]
                A[i,i+1] =-self.dt/self.dx**2/self.c[i]/self.rho[i]*self.k[i+1]

                B[i,i] = 2 - self.dt/self.dx**2/self.c[i]/self.rho[i]*(self.k[i]+self.k[i+1])
                B[i,i-1] =self.dt/self.dx**2/self.c[i]/self.rho[i]*self.k[i]
                B[i,i+1] =self.dt/self.dx**2/self.c[i]/self.rho[i]*self.k[i+1]

        p = self.dt/self.dx**2/np.max(self.c)/np.max(self.rho)*np.max(self.k)
        print('p =',p)


        for t in range(1,T):     
            bb = np.dot(B,Temperatures[:,t-1]) + b[:,t-1]+b[:,t] 
            Temperatures[:,t] = np.linalg.solve(A,bb)
        return Temperatures

class CrankNicolson1DRad:
    def __init__(self, q, T_in,k,c,rho,dt,dr,steps): 
        self.q=q
        self.k=k
        self.T_in=T_in
        self.c=c
        self.rho=rho
        self.dt=dt
        self.dr=dr
        self.steps=steps
        
        
    def solve(self):
        '''Solves the matrix equation A T^n+1 = B T^n +b'''
        N=len(self.q)
        
        A = np.zeros((N,N))
        B = np.zeros((N,N))
        b = np.zeros((N))
        alpha= np.zeros((N))
        
        T=np.zeros((N,self.steps))
        T[:,0]=self.T_in[:]
        
        for p in range(1,N-1):
            alpha[p]=(self.c[p]+self.c[p+1])*(self.rho[p]+self.rho[p+1])/2*self.dr**2/self.dt
            
            b[p]=-2*self.q[p]*self.dr**2
            
            A[p,p-1]=self.k[p]*(1-1/(2*p))
            A[p,p]=-self.k[p]*(1-1/(2*p))-self.k[p+1]*(1+1/(2*p))-alpha[p]
            A[p,p+1]=self.k[p+1]*(1+1/(2*p))
            
            B[p,p-1]=-self.k[p]*(1-1/(2*p))
            B[p,p]=+self.k[p]*(1-1/(2*p))+self.k[p+1]*(1+1/(2*p))-alpha[p]
            B[p,p+1]=-self.k[p+1]*(1+1/(2*p))
            
            
        A[0,0]= -2*self.k[1]-alpha[0]
        A[0,1]=2*self.k[1]
        B[0,0]=2*self.k[1]-alpha[0]
        B[0,1]=-2*self.k[1]
        
        A[N-1,N-1]=1
        
        B[N-1,N-1]=1           
        
        for i in range(self.steps-1):
            bb = np.dot(B,T[:,i])+b

            T[:,i+1]=np.linalg.solve(A,bb)
        return T

class PoissonFlow1DRad:
    def __init__(self, q, kappa, dr,c,rho,freq):        
        self.q=q
        self.kappa=kappa
        self.dr=dr
        self.c=c
        self.rho=rho        
        self.freq=freq
        
    
    def solve(self):
        N=len(self.q)
        A = np.zeros((N,N),dtype=complex)
        b = np.zeros((N))        
        omega = 2*np.pi*self.freq
        for p in range(1,N-1):
            A[p,p-1]= self.kappa[p]*(1-1/2/p)
            A[p,p]= (-self.kappa[p]*(1-1/2/p)-self.kappa[p+1]*(1+1/2/p)-(self.c[p+1]+self.c[p])/2*(self.rho[p+1]+self.rho[p])/2*self.dr**2*omega*1j)
            A[p,p+1]=self.kappa[p+1]*(1+1/2/p)     
        A[N-1,N-1]= 1
        A[0,0]= -2*self.kappa[p]-(self.c[p+1]+self.c[p])/2*(self.rho[p+1]+self.rho[p])/2*self.dr**2*omega*1j 
        A[0,1]= 2*self.kappa[p]
        b[:]=-self.dr**2*self.q[:]
        b[-1]=0
        temp = np.linalg.solve(A, b)
        return temp

class driftDiff2D:
    '''
    This class solves the drift diffusion equation in 2 dimensions.
    
    '''
    def __init__(self,n,f,g,q,dx,dy,dt,mu,D,zero_flux=True,iterations=10,append=1):
        self.n=n
        self.f=f
        self.g=g
        self.q=q
        self.dx=dx
        self.dy=dy
        self.dt=dt
        self.mu=mu
        self.D=D
        self.zero_flux=zero_flux
        self.iterations = iterations
        self.append = append        
        
    def cnSolve(self):
        Arow = []
        Acol = []
        Adata = []
                
        nx = self.n.shape[0]
        ny = self.n.shape[1]
        
        steps = nx*ny
        Identity = np.identity(steps)
        
        #These are independent of time
        p = self.dt*self.D/self.dx**2
        q = self.dt*self.D/self.dy**2
        r = self.dt*self.mu/self.dx/2
        s = self.dt*self.mu/self.dy/2
        
        print('nx=',nx)
        print('ny=',ny)        
        print('p=',p,'q=',q,'r=',r,'s=',s)
        print('iterations=',self.iterations)
        
        b=np.zeros((nx,ny))
        b[:,:]=self.q[:,:]*self.dt
        
        
        for i in range(nx):
            for j in range(ny):
                Adiag = j*nx+i
                
                if i ==0:
                    if j==0:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(self.f[i+1,j]+self.f[i,j])+2*s*(self.g[i,j+1]+self.g[i,j]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+nx)
                        Adata.append(-2*q)
                        
                    elif j==ny-1: 
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(self.f[i+1,j]+self.f[i,j])+2*s*(-self.g[i,j]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag-nx)
                        Adata.append(-2*q)
                        
                    else:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(self.f[i+1,j]+self.f[i,j])+s*(self.g[i,j+1]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-nx)
                        Adata.append(-q-s*self.g[i,j])
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag+1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+nx)
                        Adata.append(-q+s*self.g[i,j])
                    
                elif i == nx-1:
                    if j==0:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(-self.f[i,j]-self.f[i-1,j])+2*s*(self.g[i,j+1]-self.g[i,j]))                        
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+nx)
                        Adata.append(-2*q)
                        
                        
                    elif j==ny-1:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(-self.f[i,j]-self.f[i-1,j])+2*s*(self.g[i,j]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag-nx)
                        Adata.append(-2*q)
                        
                    else:
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(2+2*p+2*q+2*r*(-self.f[i,j]-self.f[i-1,j])+s*(self.g[i,j+1]-self.g[i,j-1]))
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-nx)
                        Adata.append(-q-s*self.g[i,j])
                        
                        Arow.append(Adiag)
                        Acol.append(Adiag-1)
                        Adata.append(-2*p)
                          
                        Arow.append(Adiag)
                        Acol.append(Adiag+nx)
                        Adata.append(-q+s*self.g[i,j])
                                            
                elif j == 0:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*p+2*q+2*r*(self.f[i+1,j]-self.f[i-1,j])+2*s*(self.g[i,j+1]+self.g[i,j]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-p-r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-p+r*self.f[i,j])
                      
                    Arow.append(Adiag)
                    Acol.append(Adiag+nx)
                    Adata.append(-2*q)
                    
                elif j == ny-1:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*p+2*q+r*(self.f[i+1,j]-self.f[i-1,j])+2*s*(-self.g[i,j]-self.g[i,j-1]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-p-r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-p+r*self.f[i,j])
                      
                    Arow.append(Adiag)
                    Acol.append(Adiag-nx)
                    Adata.append(-2*q)
                    
                else:
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(2+2*p+2*q+r*(self.f[i+1,j]-self.f[i-1,j])+s*(self.g[i,j+1]-self.g[i,j-1]))
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+1)
                    Adata.append(-p+r*self.f[i,j])
        
                    Arow.append(Adiag)
                    Acol.append(Adiag-1)
                    Adata.append(-p-r*self.f[i,j])
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag+nx)
                    Adata.append(-q+s*self.g[i,j])
        
                    Arow.append(Adiag)
                    Acol.append(Adiag-nx)
                    Adata.append(-q-s*self.g[i,j])
                    
            
        Adata_coo=coo_matrix((Adata, (Arow, Acol)), shape=(steps, steps))
        Adata_csr = Adata_coo.tocsr()
        Adata_dense = Adata_coo.todense()
        
        Bdata_dense = 4*Identity-Adata_dense  
            
        n_flat = np.zeros((steps))    
            
        for j in range(ny):
            for i in range(nx):
                n_flat[nx*j+i] = self.n[i,j]
                
        b_flat = np.zeros((1,nx*ny))
        for j in range(ny):
            for i in range(nx):
                b_flat[0,nx*j+i] = b[i,j]
        
        
        output_list = []        
              
        iteration_counter = 0
        
        for p in range(self.iterations):
            if iteration_counter%self.append==0:
                output_array = np.zeros((nx,ny))  
                for j in range(ny):
                    for i in range(nx):
                        output_array[i,j]=n_flat[j*nx+i]
                
                output_list.append(output_array)
            
            iteration_counter +=1    
            bb = Bdata_dense.dot(n_flat)
            #print(bb.shape)              
            n_flat[:] = spsolve(Adata_csr,bb.T+2*b_flat.T) 
            
        return output_list    

class driftDiff1D:
    def __init__(self,n_in,f,dx,dt,mu,D,zero_flux,iterations,append):
        self.n_in=n_in
        self.f=f
        self.dx=dx
        self.dt=dt
        self.mu=mu
        self.D=D
        self.zero_flux=zero_flux
        self.iterations = iterations
        self.append = append
        
    def ftcsSolve(self):
        
        r = self.dt*self.D/self.dx**2
        s = self.dt*self.mu/2/self.dx
        append_counter =0
        output=np.zeros((len(self.n_in),self.iterations/self.append))
        
        n = self.n_in.copy()
        for iteration in range(self.iterations):
            if iteration%self.append==0:
                output[:,append_counter]=n[:]
                append_counter += 1
            n_cp = n.copy()
            n[1:-1] = n_cp[1:-1] + r*(n_cp[2:]-2*n_cp[1:-1]+n_cp[:-2])-\
            s*n_cp[1:-1]*(self.f[2:]-self.f[:-2])-\
            s*self.f[1:-1]*(n_cp[2:]-n_cp[:-2])

            if self.zero_flux==True:
                n[0]=(1-2*r-2*s*(self.f[1]+self.f[0]))*n_cp[0]+2*r*n_cp[1]
                n[-1]=(1-2*r+2*s*(self.f[-1]+self.f[-2]))*n_cp[-1]+2*r*n_cp[-2]
            else:
                n[0]=0
                n[-1]=0            
                
        return output
        
    def cnSolve(self):
        steps=len(self.n_in)
        A=np.zeros((steps,steps))
        B=np.zeros((steps,steps))
        r = self.dt*self.D/self.dx**2
        s = self.dt*self.mu/2/self.dx
        append_counter =0
        output=np.zeros((steps,self.iterations/self.append)) 
        n = self.n_in.copy()
        
        row=[]
        col=[]
        Adata=[]

        for i in range(1,steps-1):
            Adiag=i            
            row.append(Adiag)
            col.append(Adiag)
            Adata.append(2+2*r+s*(self.f[i+1]-self.f[i-1]))
            
            row.append(Adiag)
            col.append(Adiag+1)
            Adata.append(-r+s*self.f[i])
            
            row.append(Adiag)
            col.append(Adiag-1)
            Adata.append(-r-s*self.f[i])


            B[i,i] = 2-2*r-s*(self.f[i+1]-self.f[i-1])
            B[i,i+1] = r-s*self.f[i]    
            B[i,i-1] = r+s*self.f[i]
            
        if self.zero_flux==True:
            row.append(0)
            col.append(0)
            Adata.append(2+2*r+2*s*(self.f[1]+self.f[0]))
            
            row.append(steps-1)
            col.append(steps-1)
            Adata.append(2+2*r-2*s*(self.f[-1]+self.f[-2]))
            
            row.append(steps-1)
            col.append(steps-2)
            Adata.append(-2*r)
            
            row.append(0)
            col.append(1)
            Adata.append(-2*r)
            
            B[0,0] =2-2*r-2*s*(self.f[1]+self.f[0])
            B[steps-1,steps-1] = 2-2*r+2*s*(self.f[-1]+self.f[-2])
            B[0,1] = +2*r
            B[steps-1,steps-2] = +2*r
        else:
            row.append(0)
            col.append(0)
            Adata.append(1)
            
            row.append(steps-1)
            col.append(steps-1)
            Adata.append(1)
            
            B[0,0]=0
            B[steps-1,steps-1]=0
            
        Adata_coo=coo_matrix((Adata, (row, col)), shape=(steps, steps))
        Adata_csr = Adata_coo.tocsr()
        
        for iteration in range(self.iterations):
            if iteration%self.append==0:
                output[:,append_counter]=n[:]
                append_counter += 1
            bb = B.dot(n)
            n[:] = spsolve(Adata_csr,bb)
        return output

class PoissonPot2D:
    def __init__(self,Q_in,er,V_bound,dx,dy,dirichlet):        
        self.er = er
        self.V_bound = V_bound
        self.dx=dx
        self.dy=dy
        self.Q_in =Q_in
        self.dirichlet=dirichlet
        
    
    def sparseSolve(self):
        row = []
        col = []
        data = []
        nx = self.Q_in.shape[0]
        ny = self.Q_in.shape[1]
        print('nx=',nx)
        print('ny=',ny)
        
        if self.dirichlet == True:
            for j in range(ny):
                for i in range(nx):
                    Adiag = j*nx+i
                    if j==0:
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif j==ny-1:
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif i==0:
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)    
                    elif i==nx-1:
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    else:
                        a0=-1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                        1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                        a1=1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx
                        a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                        a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                        a4=1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                        
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(a0)
                        
                        row.append(Adiag)
                        col.append(Adiag-1)
                        data.append(a1)
                        
                        row.append(Adiag)
                        col.append(Adiag+nx)
                        data.append(a2)
                        
                        row.append(Adiag)
                        col.append(Adiag+1)
                        data.append(a3)
                        
                        row.append(Adiag)
                        col.append(Adiag-nx)
                        data.append(a4)
                        
                        
                        
                        
        elif self.dirichlet == False:
            for j in range(ny):
                for i in range(nx):
                    Adiag = j*nx+i
                    if j==0:
                        if i==0:
                            #Bottom left corner
                            a0=-1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy
                            a1=1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx
                            a2=1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx
                            a4=1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy
                                                       
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)
                            
                            row.append(Adiag)
                            col.append(Adiag+nx)
                            data.append(a2+a4)
                            
                            row.append(Adiag)
                            col.append(Adiag+1)
                            data.append(a3+a1)
                            
                                                
                        elif i==nx-1:
                            #Bottom right corner
                            a0=-1/2*(self.er[i-1,j]+self.er[i-1,j])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx-\
                            1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j]+self.er[i-1,j])*self.dx/self.dy
                            a1=1/2*(self.er[i-1,j]+self.er[i-1,j])*self.dy/self.dx
                            a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx
                            a4=1/2*(self.er[i,j]+self.er[i-1,j])*self.dx/self.dy
                            
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)
                            
                            row.append(Adiag)
                            col.append(Adiag-1)
                            data.append(a1+a3)
                            
                            row.append(Adiag)
                            col.append(Adiag+nx)
                            data.append(a2+a4)
                            
                        else:
                            #Bottom edge
                            a0=-1/2*(self.er[i-1,j]+self.er[i-1,j])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx-\
                            1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j]+self.er[i-1,j])*self.dx/self.dy
                            a1=1/2*(self.er[i-1,j]+self.er[i-1,j])*self.dy/self.dx
                            a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j])*self.dy/self.dx
                            a4=1/2*(self.er[i,j]+self.er[i-1,j])*self.dx/self.dy
                            
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)
                            
                            row.append(Adiag)
                            col.append(Adiag-1)
                            data.append(a1)                            
                            
                            row.append(Adiag)
                            col.append(Adiag+nx)
                            data.append(a2+a4)
                            
                            row.append(Adiag)
                            col.append(Adiag+1)
                            data.append(a3)
                        
                    elif j==ny-1:
                        if i==0:
                            #Top left corner
                            a0=-1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j-1]+self.er[i,j-1])*self.dx/self.dy
                            a1=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                            a2=1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                            a4=1/2*(self.er[i,j-1]+self.er[i,j-1])*self.dx/self.dy
                        
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)                            
                            
                            row.append(Adiag)
                            col.append(Adiag+1)
                            data.append(a3+a1)
                            
                            row.append(Adiag)
                            col.append(Adiag-nx)
                            data.append(a4+a2)                        
                                                                                   
                        elif i==nx-1:
                            #Top right corner
                            a0=-1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                            a1=1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx
                            a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                            a4=1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                            
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)
                            
                            row.append(Adiag)
                            col.append(Adiag-1)
                            data.append(a1+a3)
                                                                                    
                            row.append(Adiag)
                            col.append(Adiag-nx)
                            data.append(a4+a2)

                        else:
                            #Top edge
                            a0=-1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                            1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                            1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                            a1=1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx
                            a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                            a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                            a4=1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                            
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(a0)
                            
                            row.append(Adiag)
                            col.append(Adiag-1)
                            data.append(a1)
                                                       
                            row.append(Adiag)
                            col.append(Adiag+1)
                            data.append(a3)
                            
                            row.append(Adiag)
                            col.append(Adiag-nx)
                            data.append(a4+a2)
                                                                                
                    elif i==0:
                        #left edge
                        a0=-1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy-\
                        1/2*(self.er[i,j-1]+self.er[i,j-1])*self.dx/self.dy
                        a1=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                        a2=1/2*(self.er[i,j]+self.er[i,j])*self.dx/self.dy
                        a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                        a4=1/2*(self.er[i,j-1]+self.er[i,j-1])*self.dx/self.dy
                        
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(a0)
                                                
                        row.append(Adiag)
                        col.append(Adiag+nx)
                        data.append(a2)
                        
                        row.append(Adiag)
                        col.append(Adiag+1)
                        data.append(a3+a1)
                        
                        row.append(Adiag)
                        col.append(Adiag-nx)
                        data.append(a4)
  
                    elif i==nx-1:
                        #Right edge
                        a0=-1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                        1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                        a1=1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx
                        a2=1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy
                        a3=1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx
                        a4=1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy
                        
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(a0)
                        
                        row.append(Adiag)
                        col.append(Adiag-1)
                        data.append(a1+a3)
                        
                        row.append(Adiag)
                        col.append(Adiag+nx)
                        data.append(a2)
                        
                        row.append(Adiag)
                        col.append(Adiag-nx)
                        data.append(a4)
                        
                    else:
                        #Middle
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(-1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx-\
                        1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy-\
                        1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy)
                        
                        row.append(Adiag)
                        col.append(Adiag-1)
                        data.append(1/2*(self.er[i-1,j]+self.er[i-1,j-1])*self.dy/self.dx)
                        
                        row.append(Adiag)
                        col.append(Adiag+nx)
                        data.append(1/2*(self.er[i-1,j]+self.er[i,j])*self.dx/self.dy)
                        
                        row.append(Adiag)
                        col.append(Adiag+1)
                        data.append(1/2*(self.er[i,j]+self.er[i,j-1])*self.dy/self.dx)
                        
                        row.append(Adiag)
                        col.append(Adiag-nx)
                        data.append(1/2*(self.er[i,j-1]+self.er[i-1,j-1])*self.dx/self.dy)
                   
                
        As=coo_matrix((data, (row, col)), shape=(nx*ny, nx*ny))
        As = As.tolil()        
                
        Q_in_flat = np.zeros(nx*ny)
        for j in range(ny):
            for i in range(nx):
                Q_in_flat[nx*j+i] = self.Q_in[i,j]        
                
        BC_mask_flat=np.zeros(nx*ny)        
                
        for BC in self.V_bound:
            for j in range(ny):
                for i in range(nx):
                    BC_mask_flat[nx*j+i] = BC[0][i,j]
            
            for k in range(nx*ny):
                if BC_mask_flat[k]==1:
                    Q_in_flat[k] = -BC[1]
                    As[k,k]=1
                    As[k,k-1]=0
                    As[k,k+1]=0
                    As[k,k+nx]=0
                    As[k,k-nx]=0
                    
        #print(Q_in_flat)    
        As = As.tocsr()       
        #print('A shape is',As.shape)        
        #print('Q shape is',Q_in_flat.shape)
        n = np.zeros(nx*ny)
        n = spsolve(As,-Q_in_flat)
            
        unwrapped = np.zeros((nx,ny))
        for j in range(ny):
            for i in range(nx):
                unwrapped[i,j]= n[nx*j+i]
        
        return unwrapped
        
class PoissonFlow2D:
    '''
    Solves the advection diffusion equation with a time-independent or sinusoidal heat source. The boundary conditions
    are Dirichlet by default but a Neumann condition can be included at the top edge. Velocity is only in the x-direction.
    A staggered grid is used for the thermal conductivity, which can vary over the domain. A variable dy can be included
    by passing an array with length ny (number of grid points in y) into the solver.

    Parameters
    ----------
    q np.ndarray: heat source density (W/m2)
    kappa np.ndarray: thermal conductivity (W/m/K)
    v np.ndarray: x component of velocity (m/s)
    dx float: spacing between grid points in x (m)
    dy np.ndarray: spacing between grid points in y (m)
    rho np.ndarray: density array (kg/m3)
    freq float: frequency (Hz) of the heat source. If 0, then dc heat source
    neu_top boolean: if True, use a Neumann zero condition at the top, otherwise Dirichlet zero condition

    Returns
    ----------
    unwrapped np.ndarray(complex): the complex 2d temperature field

    '''

    def __init__(self, q, kappa, v, dx, dy, c, rho, freq, neu_top, neu_left = False, neu_right = False, cg = False):
        self.q=q
        self.v=v
        self.dx=dx

        self.c=c
        self.rho=rho        
        self.freq=freq
        self.neu_top = neu_top
        self.neu_left = neu_left
        self.neu_right = neu_right
        self.cg = cg

        self.nx = kappa.shape[0]
        self.ny = kappa.shape[1]

        # There is a nx,ny grid of temperature. Since we use a staggered grid of thermal conductivity, the array size
        # needs to be nx+1, ny+1. First the -2 col is copied to the -1 col. Then the -2 row is copied to the -1 row.
        #print(kappa)
        self.kappa = np.zeros((self.nx + 1, self.ny + 1))
        self.kappa[1:,1:] = kappa[:,:]
        self.kappa[0,1:] = self.kappa[1,1:]
        self.kappa[:,0] = self.kappa[:,1]
        #print(self.kappa)

        #
        self.dy = np.zeros(self.ny + 2)
        self.dy[1:-1] = dy[:]
        self.dy[0] = self.dy[1]
        self.dy[-1] = self.dy[-2]
        
        
    def make_matrix(self):
            row = []
            col = []
            data = []


            for j in range(self.ny):
                for i in range(self.nx):
                    Adiag = j*self.nx+i

                    # left
                    # previously
                    #a1 = 0.5 * (self.k[i - 1, j] + self.k[i - 1, j - 1]) * self.dy[j] / self.dx - \
                    #     0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]
                    a1 = 1 / 2 * (self.kappa[i, j+1] + self.kappa[i, j]) * self.dy[j+1] / self.dx - \
                         1 / 2 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j+1]
                    # top
                    # previously
                    # a2 = (self.k[i - 1, j] + self.k[i, j]) * self.dx / (self.dy[j] + self.dy[j])
                    a2 = (self.kappa[i, j + 1] + self.kappa[i + 1, j + 1]) * self.dx / (self.dy[j+2] + self.dy[j+1])

                    # right
                    # previously
                    #a3 = 0.5 * (self.k[i, j] + self.k[i, j - 1]) * self.dy[j] / self.dx + \
                    #0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]
                    a3 = 1 / 2 * (self.kappa[i + 1, j + 1] + self.kappa[i + 1, j]) * self.dy[j + 1] / self.dx + \
                         1 / 2 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j+1]
                    # bottom
                    # previously
                    # a4 = (self.k[i, j - 1] + self.k[i - 1, j - 1]) * self.dx / (self.dy[j] + self.dy[j - 1])
                    a4 = (self.kappa[i+1, j ] + self.kappa[i, j]) * self.dx / (
                            self.dy[j+1] + self.dy[j])

                    a0 = - a1 - a2 - a3 - a4 + \
                         self.c[i, j] * self.rho[i, j] * self.freq * np.pi * 2 * self.dx * self.dy[j+1] * 1j

                    if j == 0:
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)

                    elif i == 0 and j == self.ny - 1:
                        if self.neu_left == False:
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(1)
                        else:
                            data.append(a0)
                            row.append(Adiag)
                            col.append(Adiag)

                            #data.append(a1 + a3)
                            #row.append(Adiag)
                            #col.append(Adiag - 1)

                            #data.append(a2)
                            #row.append(Adiag)
                            #col.append(Adiag + self.nx)

                            data.append(a3+a1)
                            row.append(Adiag)
                            col.append(Adiag + 1)

                            data.append(a2 + a4)
                            row.append(Adiag)
                            col.append(Adiag - self.nx)

                    elif i == 0:
                        if self.neu_left == False:
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(1)
                        else:
                            data.append(a0)
                            row.append(Adiag)
                            col.append(Adiag)

                            #data.append(a1 + a3)
                            #row.append(Adiag)
                            #col.append(Adiag - 1)

                            data.append(a2)
                            row.append(Adiag)
                            col.append(Adiag + self.nx)

                            data.append(a3+a1)
                            row.append(Adiag)
                            col.append(Adiag + 1)

                            data.append(a4)
                            row.append(Adiag)
                            col.append(Adiag - self.nx)

                    elif i == self.nx - 1 and j == self.ny -1:
                        if self.neu_right == False:
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(1)

                        else:
                            data.append(a0)
                            row.append(Adiag)
                            col.append(Adiag)

                            data.append(a1+a3)
                            row.append(Adiag)
                            col.append(Adiag - 1)

                            #data.append(a2)
                            #row.append(Adiag)
                            #col.append(Adiag + self.nx)

                            #data.append(a3 + a1)
                            #row.append(Adiag)
                            #col.append(Adiag + 1)

                            data.append(a2 + a4)
                            row.append(Adiag)
                            col.append(Adiag - self.nx)

                    elif i == self.nx - 1:
                        if self.neu_right == False:
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(1)

                        else:
                            data.append(a0)
                            row.append(Adiag)
                            col.append(Adiag)

                            data.append(a1+a3)
                            row.append(Adiag)
                            col.append(Adiag - 1)

                            data.append(a2)
                            row.append(Adiag)
                            col.append(Adiag + self.nx)

                            #data.append(a3 + a1)
                            #row.append(Adiag)
                            #col.append(Adiag + 1)

                            data.append(a4)
                            row.append(Adiag)
                            col.append(Adiag - self.nx)

                    elif j == self.ny - 1:
                        if self.neu_top == False:
                            row.append(Adiag)
                            col.append(Adiag)
                            data.append(1)
                        else:
                            data.append(a0)
                            row.append(Adiag)
                            col.append(Adiag)

                            data.append(a1)
                            row.append(Adiag)
                            col.append(Adiag - 1)

                            data.append(a3)
                            row.append(Adiag)
                            col.append(Adiag + 1)

                            data.append(a2 + a4)
                            row.append(Adiag)
                            col.append(Adiag - self.nx)

                    else:
                        data.append(a0)
                        row.append(Adiag)
                        col.append(Adiag)
                        
                        data.append(a1)
                        row.append(Adiag)
                        col.append(Adiag-1)
                        
                        data.append(a2)
                        row.append(Adiag)
                        col.append(Adiag+self.nx)
                        
                        data.append(a3)
                        row.append(Adiag)
                        col.append(Adiag+1)
                        
                        data.append(a4)
                        row.append(Adiag)
                        col.append(Adiag-self.nx)
                                          
            As=coo_matrix((data, (row, col)), shape=(self.nx*self.ny, self.nx*self.ny))
            self.As = As.tocsr()

    def solve(self):
        Q_in_flat = np.zeros(self.nx*self.ny)
        for j in range(self.ny):
            for i in range(self.nx):
                Q_in_flat[self.nx*j+i] = self.q[i,j]*self.dx*self.dy[j+1]
            

        n = np.zeros(self.nx*self.ny,dtype=complex)

        if self.cg == False:
            n = spsolve(self.As,-Q_in_flat)
        elif self.cg == True:
            n, exit_code = cg(self.As, -Q_in_flat)
      
        unwrapped = np.zeros((self.nx,self.ny),dtype=complex)
        for j in range(self.ny):
            for i in range(self.nx):
                unwrapped[i,j]= n[self.nx*j+i]
        
        return unwrapped




class PoissonFlow2DCN:
    '''
    Solves the time dependent diffusion equation via the Crank-Nicolson method with a materials stack and variable dy.

    There is a staggered grid. Each grid poitn
    '''
    def __init__(self, qs, k, rho, c, v, u0, T, nt, dx, dy, nx, ny, neu_top, cg = False):
        """
        Initialize the object with the given parameters.

        Parameters
        ----------
        qs (list of np.ndarray): Heat source density (W/m2).
        k (np.ndarray): Thermal conductivity (W/m/K).
        rho (np.ndarray): Density (kg/m3).
        c (np.ndarray): Specific heat capacity (J/K/m3).
        v (np.ndarray): Velocity in the x-direction (m/s).
        u0 (np.ndarray): Initial temperature condition (K).
        T (float): Total time (s).
        nt (int): Number of time steps.
        dx (float): Grid spacing along the x-axis (m).
        dy (float): Grid spacing along the y-axis (m).
        nx (int): Number of grid points along the x-axis.
        ny (int): Number of grid points along the y-axis.
        """

        self.qs = qs
        self.k = k
        self.rho = rho
        self.c = c
        self.v = v
        self.u0 = u0

        self.T = T
        self.nt = nt
        self.dt = T/(nt-1)
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.neu_top = neu_top
        self.cg = cg

    def make_matricies(self):
        '''
        Solves the previously created object using the Crank-Nicolson method.

        Returns
        -------
        us_wrap (list of np.npdarray): The temperature fields for each timestep.
        '''

        # Build the sparse matrix lists
        Arow = []
        Acol = []
        Adata = []

        Brow = []
        Bcol = []
        Bdata = []

        for j in range(self.ny):
            for i in range(self.nx):
                Adiag = j * self.nx + i

                if j == 0:
                    #self.q[i, j] = 0
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(1)

                    Brow.append(Adiag)
                    Bcol.append(Adiag)
                    Bdata.append(1)

                elif i == 0:
                    #self.q[i, j] = 0
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(1)

                    Brow.append(Adiag)
                    Bcol.append(Adiag)
                    Bdata.append(1)

                elif i == self.nx - 1:
                    #self.q[i, j] = 0
                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(1)

                    Brow.append(Adiag)
                    Bcol.append(Adiag)
                    Bdata.append(1)

                elif j == self.ny - 1:
                    if self.neu_top == False:
                        #self.q[i, j] = 0
                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(1)

                        Brow.append(Adiag)
                        Bcol.append(Adiag)
                        Bdata.append(1)

                    elif self.neu_top == True:
                        # N.B. can't put this at top of i,j loop as indicies go out of range.
                        # u_{i-1,j} term
                        a1 = 0.5 * (self.k[i - 1, j] + self.k[i - 1, j - 1]) * self.dy[j] / self.dx - \
                             0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]

                        # u_{i,j+1} term - N.B. the self.dy[j + 1] index goes out of range so assuming = self.dy[j]
                        a2 = (self.k[i - 1, j] + self.k[i, j]) * self.dx / (self.dy[j] + self.dy[j])

                        # u_{i+1,j} term
                        a3 = 0.5 * (self.k[i, j] + self.k[i, j - 1]) * self.dy[j] / self.dx + \
                             0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]

                        # u_{i,j-1} term
                        a4 = (self.k[i, j - 1] + self.k[i - 1, j - 1]) * self.dx / (self.dy[j] + self.dy[j - 1])

                        # u
                        a5 = 2 * 0.25 * (self.c[i, j - 1] * self.rho[i, j - 1] + self.c[i, j] * self.rho[i, j] +
                                         self.c[i - 1, j - 1] * self.rho[i - 1, j - 1] + self.c[i - 1, j] * self.rho[
                                             i - 1, j]) / self.dt * self.dx * self.dy[j]

                        Arow.append(Adiag)
                        Acol.append(Adiag)
                        Adata.append(- a1 - a2 - a3 - a4 - a5)

                        Brow.append(Adiag)
                        Bcol.append(Adiag)
                        Bdata.append(a1 + a2 + a3 + a4 - a5)

                        Arow.append(Adiag)
                        Acol.append(Adiag - 1)
                        Adata.append(a1)

                        Brow.append(Adiag)
                        Bcol.append(Adiag - 1)
                        Bdata.append(-a1)

                        Arow.append(Adiag)
                        Acol.append(Adiag + 1)
                        Adata.append(a3)

                        Brow.append(Adiag)
                        Bcol.append(Adiag + 1)
                        Bdata.append(-a3)

                        #Arow.append(Adiag)
                        #Acol.append(Adiag + self.nx)
                        #Adata.append(a2)

                        #Brow.append(Adiag)
                        #Bcol.append(Adiag + self.nx)
                        #Bdata.append(-a2)

                        Arow.append(Adiag)
                        Acol.append(Adiag - self.nx)
                        Adata.append(a2 + a4)

                        Brow.append(Adiag)
                        Bcol.append(Adiag - self.nx)
                        Bdata.append(-a2 - a4)

                else:
                    # N.B. can't put this at top of i,j loop as indicies go out of range.
                    # u_{i-1,j} term
                    a1 = 0.5 * (self.k[i - 1, j] + self.k[i - 1, j - 1]) * self.dy[j] / self.dx - \
                         0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]

                    # u_{i,j+1} term
                    a2 = (self.k[i - 1, j] + self.k[i, j]) * self.dx / (self.dy[j] + self.dy[j + 1])

                    # u_{i+1,j} term
                    a3 = 0.5 * (self.k[i, j] + self.k[i, j - 1]) * self.dy[j] / self.dx + \
                         0.5 * self.c[i, j] * self.rho[i, j] * self.v[i, j] * self.dy[j]

                    # u_{i,j-1} term
                    a4 = (self.k[i, j - 1] + self.k[i - 1, j - 1]) * self.dx / (self.dy[j] + self.dy[j - 1])

                    # u
                    a5 = 2 * 0.25 * (self.c[i, j - 1] * self.rho[i, j - 1] + self.c[i, j] * self.rho[i, j] +
                                     self.c[i - 1, j - 1] * self.rho[i - 1, j - 1] + self.c[i - 1, j] * self.rho[
                                         i - 1, j]) / self.dt * self.dx * self.dy[j]

                    Arow.append(Adiag)
                    Acol.append(Adiag)
                    Adata.append(- a1 - a2 - a3 - a4 - a5)

                    Brow.append(Adiag)
                    Bcol.append(Adiag)
                    Bdata.append(a1 + a2 + a3 + a4 - a5)

                    Arow.append(Adiag)
                    Acol.append(Adiag - 1)
                    Adata.append(a1)

                    Brow.append(Adiag)
                    Bcol.append(Adiag - 1)
                    Bdata.append(-a1)
                    
                    Arow.append(Adiag)
                    Acol.append(Adiag + 1)
                    Adata.append(a3)

                    Brow.append(Adiag)
                    Bcol.append(Adiag + 1)
                    Bdata.append(-a3)

                    Arow.append(Adiag)
                    Acol.append(Adiag + self.nx)
                    Adata.append(a2)

                    Brow.append(Adiag)
                    Bcol.append(Adiag + self.nx)
                    Bdata.append(-a2)

                    Arow.append(Adiag)
                    Acol.append(Adiag - self.nx)
                    Adata.append(a4)

                    Brow.append(Adiag)
                    Bcol.append(Adiag - self.nx)
                    Bdata.append(-a4)

        # Create the matricies from the lists
        A_coo = coo_matrix((Adata, (Arow, Acol)), shape=(self.nx * self.ny, self.nx * self.ny))
        A_csr = A_coo.tocsr()
        #A_dense = A_coo.todense()
        self.A_csr = A_csr

        B_coo = coo_matrix((Bdata, (Brow, Bcol)), shape=(self.nx * self.ny, self.nx * self.ny))
        B_coo = B_coo.tocsr()
        B_dense = B_coo.todense()
        self.B_dense = B_dense

    def cn_solve(self):
        # Flatten the q arrays and integrate
        Qs = []
        for q in self.qs:
            Q_flat = np.zeros(self.nx * self.ny)
            for j in range(self.ny):
                for i in range(self.nx):
                    Q_flat[self.nx * j + i] = q[i, j]*self.dx*self.dy[j]
            Qs.append(Q_flat)
        #print(Q_flat.shape)

        # Flatten the starting condition
        u_flat = np.zeros(self.nx * self.ny)
        for j in range(self.ny):
            for i in range(self.nx):
                u_flat[self.nx * j + i] = self.u0[i, j]

        us = [u_flat]

        # The main loop
        for k in range(1, self.nt):
            bb = self.B_dense.dot(us[k-1]) - Qs[k - 1] - Qs[k]
            if self.cg:
                uu, exit_code = cg(self.A_csr, bb.T)
            else:
                uu = spsolve(self.A_csr,bb.T)
            us.append(uu)

        # Wrap up the results
        us_wrap = []
        for u in us:
            u_temp = np.zeros((self.nx, self.ny))
            for j in range(self.ny):
                for i in range(self.nx):
                    u_temp[i, j] = u[self.nx * j + i]
            us_wrap.append(u_temp)

        return us_wrap


class PoissonFlow3D:
    def __init__(self, Q_in, kappa, v_in, dx,dy,dz,c,rho):        
        self.Q_in=Q_in
        self.kappa=kappa
        self.v_in=v_in
        self.dx=dx
        self.dy=dy
        self.dz=dz
        self.c=c
        self.rho=rho
        

    def sparseSolve(self):
        row = []
        col = []
        data = []
        nx = self.kappa.shape[0]
        ny = self.kappa.shape[1]
        nz = self.kappa.shape[2]
        print('nx=',nx)
        print('ny=',ny)
        print('nz=',nz)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    Adiag = k*nx*ny+j*nx+i
                    if k==0:
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif k==(nz-1):
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif j==0:   
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif j==(ny-1):
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif i==0:
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    elif i==(nx-1):
                        self.Q_in[i,j,k] = 0
                        row.append(Adiag)
                        col.append(Adiag)
                        data.append(1)
                    else:
                        data.append(-1/4*(self.kappa[i-1,j,k]+self.kappa[i-1,j-1,k]+self.kappa[i-1,j,k-1]+self.kappa[i-1,j-1,k-1])*self.dy*self.dz/self.dx-\
                        1/2*self.c*self.rho*self.v_in[i,j,k]*self.dy*self.dz-\
                        1/4*(self.kappa[i,j,k]+self.kappa[i,j-1,k]+self.kappa[i,j,k-1]+self.kappa[i,j-1,k-1])*self.dy*self.dz/self.dx+\
                        1/2*self.c*self.rho*self.v_in[i,j,k]*self.dy*self.dz-\
                        1/4*(self.kappa[i,j-1,k]+self.kappa[i-1,j-1,k]+self.kappa[i,j-1,k-1]+self.kappa[i-1,j-1,k-1])*self.dx*self.dz/self.dy-\
                        1/4*(self.kappa[i,j,k]+self.kappa[i-1,j,k]+self.kappa[i,j,k-1]+self.kappa[i-1,j,k-1])*self.dx*self.dz/self.dy-\
                        1/4*(self.kappa[i,j,k-1]+self.kappa[i-1,j,k-1]+self.kappa[i,j-1,k-1]+self.kappa[i-1,j-1,k-1])*self.dx*self.dy/self.dz-\
                        1/4*(self.kappa[i,j,k]+self.kappa[i-1,j,k]+self.kappa[i,j-1,k]+self.kappa[i-1,j-1,k])*self.dx*self.dy/self.dz)
                        row.append(Adiag)
                        col.append(Adiag)
                        
                        data.append(1/4*(self.kappa[i-1,j,k]+self.kappa[i-1,j-1,k]+self.kappa[i-1,j,k-1]+self.kappa[i-1,j-1,k-1])*self.dy*self.dz/self.dx-\
                        1/2*self.c*self.rho*self.v_in[i,j,k]*self.dy*self.dz)
                        row.append(Adiag)
                        col.append(Adiag-1)
                        
                        data.append(1/4*(self.kappa[i,j,k]+self.kappa[i,j-1,k]+self.kappa[i,j,k-1]+self.kappa[i,j-1,k-1])*self.dy*self.dz/self.dx+\
                        1/2*self.c*self.rho*self.v_in[i,j,k]*self.dy*self.dz)
                        row.append(Adiag)
                        col.append(Adiag+1)
                        
                        data.append(1/4*(self.kappa[i,j-1,k]+self.kappa[i-1,j-1,k]+self.kappa[i,j-1,k-1]+self.kappa[i-1,j-1,k-1])*self.dx*self.dz/self.dy)
                        row.append(Adiag)
                        col.append(Adiag-nx)
                        
                        data.append(1/4*(self.kappa[i,j,k]+self.kappa[i-1,j,k]+self.kappa[i,j,k-1]+self.kappa[i-1,j,k-1])*self.dx*self.dz/self.dy)
                        row.append(Adiag)
                        col.append(Adiag+nx)
                        
                        data.append(1/4*(self.kappa[i,j,k-1]+self.kappa[i-1,j,k-1]+self.kappa[i,j-1,k-1]+self.kappa[i-1,j-1,k-1])*self.dx*self.dy/self.dz)
                        row.append(Adiag)
                        col.append(Adiag-nx*ny)
                        
                        data.append(1/4*(self.kappa[i,j,k]+self.kappa[i-1,j,k]+self.kappa[i,j-1,k]+self.kappa[i-1,j-1,k])*self.dx*self.dy/self.dz)
                        row.append(Adiag)
                        col.append(Adiag+nx*ny)
                        
        #plt.pcolor(A)               
        Q_in_flat = np.zeros(nx*ny*nz)
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    Q_in_flat[ny*nx*k+nx*j+i] = self.Q_in[i,j,k]
        
        col = np.array(col)
        row=np.array(row)
        data = np.array(data) 
        
        As=coo_matrix((data, (row, col)), shape=(nx*ny*nz, nx*ny*nz))
        As = As.tocsr()
        
        n=np.zeros(nx*ny*nz)
        n = spsolve(As,-Q_in_flat)
        unwrapped = np.zeros((nx,ny,nz))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    unwrapped[i,j,k]= n[ny*nx*k+nx*j+i]
        
        return unwrapped
        

 
        
