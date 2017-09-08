"""
Dylan S
Numerical and Scientific Methods
Day 22
Python 3
"""
# Content under Creative Commons Attribution license CC-BY 4.0, code under
# MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David
# Ketcheson's pendulum lesson, also under CC-BY.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def rho_red_light(nx, rho_max, rho_in):
    """Computes "red light" initial condition with shock

    Parameters
    ----------
    nx        : int
        Number of grid points in x
    rho_max   : float
        Maximum allowed car density
    rho_in    : float
        Density of incoming cars

    Returns
    -------
    rho: array of floats
        Array with initial values of density
    """
    rho = rho_max*np.ones(nx)
    rho[:int((nx-1)*3./4.)] = rho_in
    return rho

# Basic initial condition parameters
nx = 81
nt = 30
dx = 4.0/(nx-1)

rho_in = 8.
rho_max = 10.

u_max = 1.

x = np.linspace(0, 100, nx)

rho = rho_red_light(nx, rho_max, rho_in)


def computeF(u_max, rho_max, rho):
    """Computes flux F=V*rho

    Parameters
    ----------
    u_max  : float
        Maximum allowed velocity
    rho    : array of floats
        Array with density of cars at every point x
    rho_max: float
        Maximum allowed car density

    Returns
    -------
    F : array
        Array with flux at every point x
    """
    return u_max*rho*(1-rho/rho_max)



def animate(data):
    x = np.linspace(0, 4, nx)
    y = data
    line.set_data(x, y)
    return line,


def maccormack(rho, nt, dt, dx, u_max, rho_max):
    """ Computes the solution with MacCormack scheme

    Parameters
    ----------
    rho    : array of floats
            Density at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    rho_max: float
            Maximum allowed car density
    u_max  : float
            Speed limit

    Returns
    -------
    rho_n : array of floats
            Density after nt time steps at every point x
    """

    rho_n = np.zeros((nt, len(rho)))
    rho_star = np.empty_like(rho)
    rho_n[:, :] = rho.copy()
    rho_star = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho_max, rho)
        rho_star[:-1] = rho[:-1] - dt/dx * (F[1:]-F[:-1])
        Fstar = computeF(u_max, rho_max, rho_star)
        rho_n[t, 1:] = .5 * (rho[1:]+rho_star[1:] -
                             dt/dx * (Fstar[1:] - Fstar[:-1]))
        rho = rho_n[t].copy()

    return rho_n

def ftbs(rho, nt, dt, dx, rho_max, u_max):
    """ Computes the solution with forward in time, backward in space
    """
    rho_n = np.zeros((nt, len(rho)))
    rho_star = np.empty_like(rho)
    rho_n[:, :] = rho.copy()
    rho_star = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho_max, rho)
        rho_star[1:] = rho[1:] - dt/dx * (F[1:]-F[:-1])
        Fstar = computeF(u_max, rho_max, rho_star)
        rho_n[t, :-1] = .5 * (rho[:-1]+rho_star[:-1] -
                             dt/dx * (Fstar[1:] - Fstar[:-1]))
        rho = rho_n[t].copy()

    return rho_n

rho = rho_red_light(nx, rho_max, rho_in)
sigma = 1.0
dt = sigma*dx/u_max

rho_n = maccormack(rho, nt, dt, dx, u_max, rho_max)
#rho_n = ftbs(rho, nt, dt, dx, rho_max, u_max)

fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(4.5, 11.), xlabel=('Distance'),
              ylabel=('Traffic density'))
line, = ax.plot([], [], color='#ffc400', lw=2)

anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=150)


plt.show()
