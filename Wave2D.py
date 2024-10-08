import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N=N
        xi = np.linspace(0, 1, N+1)  # x coordinates
        yj = np.linspace(0, 1, N+1)  # y coordinates
        self.xij, self.yij = np.meshgrid(xi, yj, indexing='ij')
        return self.xij, self.yij

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return np.pi * np.sqrt(self.mx**2+self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        self.mx=mx
        self.my=my
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        self.create_mesh(N)
        self.mx, self.my = mx, my
        U_n = np.zeros((N+1))
        U_nm1 = np.sin(mx * np.pi * self.xij) * np.sin(my * np.pi * self.yij)
        U_n[:] = U_nm1[:] + 0.5*(self.c*self.dt)**2*(self.D @ U_nm1 + U_nm1 @ self.D.T)
        self.U_np1 = np.zeros_like(self.U_n)
        return self.U_n, self.U_nm1

    @property
    def dt(self):
        """Return the time step"""
        h = 1 / self.N  # Spatial step size
        self.h=h
        return self.cfl * h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact = np.sin(self.mx * np.pi * self.xij) * np.sin(self.my * np.pi * self.yij) * np.cos(self.w * t0)
        e2_sum= np.sum((u-u_exact)**2)
        return np.sqrt(self.h**2*e2_sum)

    def apply_bcs(self):
        self.U_n[0, :] = 0
        self.U_n[-1, :] = 0
        self.U_n[:, 0] = 0
        self.U_n[:, -1] = 0

        self.U_nm1[0, :] = 0
        self.U_nm1[-1, :] = 0
        self.U_nm1[:, 0] = 0
        self.U_nm1[:, -1] = 0

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
        self.c = c
        self.cfl = cfl
        self.initialize(N, mx, my)
        dt = self.dt
        h=1/N
        self.h=h
        xij, yij = self.create_mesh(N, N)
        D=self.D2(N)/h**2
        plotdata = {0: self.U_nm1.copy()}
        
        if store_data == 1:
            plotdata[1] = self.U_n.copy()
        for n in range(1, Nt):
            self.U_np1[:] = 2*self.U_n - self.U_nm1 + (c*dt)**2*(D @ self.U_n + self.U_n @ D.T)
            self.apply_bcs()
            self.U_nm1[:]=self.U_n
            self.U_n[:]=self.U_np1
            if n % store_data == 0 and store_data > 0:
                plotdata[n] = self.U_n.copy() # Unm1 is now swapped to Un
        if store_data == -1:
            l2_err = self.l2_error(self.U_n, Nt * dt)
            return h, l2_err       
        return xij, yij, plotdata
            


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

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :3] = -2, 2, 0
        D[-1, -3:] = 0, 2, -2
        return D

    def ue(self, mx, my):
        self.mx=mx
        self.my=my
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        self.U_n[0, :] = self.U_n[1, :]      # At x = 0
        self.U_n[-1, :] = self.U_n[-2, :]    # At x = N
        # Neumann condition on bottom and top boundaries (y = 0 and y = N)
        self.U_n[:, 0] = self.U_n[:, 1]      # At y = 0
        self.U_n[:, -1] = self.U_n[:, -2]    # At y = N

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    raise NotImplementedError
