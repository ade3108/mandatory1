import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        self.L = 1
        self.N = N
        self.h = self.L/self.N
        x = self.x = np.linspace(0, self.L, self.N+1)
        y = self.y = np.linspace(0, self.L, self.N+1) 
        self.xij, self.yij = np.meshgrid(x,y, indexing = "ij")
        return self.h

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0,:4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx * sp.pi
        ky = self.my * sp.pi
        w = self.c * sp.sqrt(kx**2 + ky**2)
        return w

    def ue(self, mx, my, x, y, t):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.N = N 
        self.mx = mx
        self.my = my

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue_f = sp.lambdify((x,y), self.ue(self.mx, self.my, x, y, t0))(self.xij, self.yij)
        uerr = ue_f - u 
        return np.sqrt(self.h**2*np.sum(uerr**2))

    def apply_bcs(self, u=None):
        u[0] = 0 
        u[-1] = 0 
        u[:,0] = 0 
        u[:,-1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N 
        self.Nt = Nt
        self.cfl = cfl 
        self.c = c
        self.mx = mx 
        self.my = my 
        self.h = self.create_mesh(self.N)
        Unp1, Un, Unm1 = np.zeros((3, self.N+1, self.N+1))
        D = (1/self.h**2)*self.D2()
        Unm1[:] = sp.lambdify((x,y), self.ue(mx, my, x, y, 0))(self.xij, self.yij)
        Un[:] = sp.lambdify((x,y), self.ue(mx, my, x, y, self.dt))(self.xij, self.yij)
        
        plotdata = {0: Unm1.copy()}
        
        for n in range(1,self.Nt+1):
            Unp1[:] = 2*Un - Unm1 + (self.c*self.dt)**2*(D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                plotdata[n] = Unm1.copy()
        if store_data == -1:
            return (self.h, [self.l2_error(Un, (self.Nt+1)*self.dt)])
        elif store_data > 0: 
            return self.xij, self.yij, plotdata

            


    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0,:4] = -2, 2, 0, 0 
        D[-1, -4:] = 0, 0, 2, -2
        return D

    def ue(self, mx, my, x, y, t):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    solD = Wave2D()
    solN = Wave2D_Neumann()
    hD, errorD = solD(N=10, Nt=10, cfl = 1/np.sqrt(2))
    hH, errorH = solN(N=10, Nt=10, cfl = 1/np.sqrt(2))
    assert abs(errorD[-1])<1e-12
    assert abs(errorH[-1])<1e-12
