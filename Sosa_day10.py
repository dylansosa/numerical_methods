# -*- coding: utf-8 -*-
"""
Dylan!
Numerical and Scientific Methods
Day 10
Python 3 & Python 2.7 compatible
"""

# Content under Creative Commons Attribution license CC-BY 4.0, code under
# MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David
# Ketcheson's pendulum lesson, also under CC-BY.

from math import sin, cos, log, ceil, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters:
g = 9.8      # gravity in m s^{-2}
v_t = 4.9   # trim velocity in m s^{-1} for a paper airplane
C_D = 1.0/5.0  # drag coefficient --- or F_D/F_L if C_L=1
C_L = 1.0   # for convenience, use C_L = 1

# set initial conditions
######eddit v0 and theta0
v0 = 14.9    # start at the trim velocity and vary from there.
theta0 = 6.074 # initial angle of trajectory
x0 = 0.0     # horizotal position is arbitrary
y0 = 2.0  # initial altitude of paper airplane

#vdot = .gsintheta - g(cd/cl)*v**2/vt**2
#thetadot = -g/v*cos(theta)g/vt**2*v



T = 20                          # final time
dt = 0.01                           # time increment
N = int(T/dt) + 1                  # number of time-steps
t = np.linspace(0, T, N)      # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 4))
u[0] = np.array([v0, theta0, x0, y0])  # fill 1st element with init. values


def f(u):
    """Returns the right-hand side of the phugoid system of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    v = u[0]
    theta = u[1]
    x = u[2]
    y = u[3]

    u_dot = np.empty_like(u)

    u_dot[0] = -g*sin(theta) - C_D/C_L * g / v_t**2 * v**2
    u_dot[1] = -g*cos(theta) / v + g / v_t**2 * v
    u_dot[2] = v*cos(theta)
    u_dot[3] = v*sin(theta)

    return u_dot

###########
def RungeKutta(u, f, dt):
    """
    Return 4th order solution using Runge Kutta method
    """
    k1 = f(u)
    k2 = f(u + (dt/2)*k1)
    k3 = f(u + (dt/2)*k2)
    k4 = f(u + dt*k3)

    return u+(dt/6)*(k1+k2+k3+k4)
###########

def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
#
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f(u) = u_dot
    dt : float
        time-increment.
#
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
#
    return u + dt * f(u)  # Note how we are able to call another function f(u)
#

# time loop - Euler method
for n in range(N-1):
    if u[n, 3] < 0:
        print('When the initial velocity is {:.2f}'.format(v0), 'meters per'
              'second and the initial angle is {:.3f}'.format(theta0),
              'radians'"\n"
              'the plane went a distance of {:.2f}'.format(u[n, 2]),
              'meters before hitting the ground at t ={:.2f}'.format(n*dt),
              'seconds')
        break
    u[n+1] = RungeKutta(u[n], f, dt)


# get the glider's position with respect to the time
x = u[:n, 2]
y = u[:n, 3]


time = n*dt
# visualization of the path
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'x', fontsize=18)
plt.ylabel(r'y', fontsize=18)
plt.title('Glider trajectory, flight time = %.2f' % time, fontsize=18)
plt.plot(x, y, 'k-', lw=2)

# Determine the convergence of the numerical solution
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

u_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):
    N = int(T/dt) + 1    # number of time-steps

    # discretize the time t
    t = np.linspace(0.0, T, N)

    # initialize the array containing the solution for each time-step
    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N-1):
        u[n+1] = euler_step(u[n], f, dt)   # call euler_step()

    # store the value of u related to one grid
    u_values[i] = u


def get_diffgrid(u_current, u_fine, dt):
    """Returns the difference between one grid and the fine one using L-1 norm.

    Parameters
    ----------
    u_current : array of float
        solution on the current grid.
    u_finest : array of float
        solution on the fine grid.
    dt : float
        time-increment on the current grid

    Returns
    -------
    diffgrid : float
        difference computed in the L-1 norm.
    """

    N_current = float(len(u_current[:, 0]))
    N_fine = float(len(u_fine[:, 0]))

    #grid_size_ratio = int(ceil((N_fine-1)/(N_current-1)))
    grid_size_ratio = int(ceil(N_fine/N_current))

    diffgrid = dt * np.sum(np.abs(u_current[:, 2]
                                  - u_fine[::grid_size_ratio, 2]))

    return diffgrid


# compute difference between one grid solution and the finest one
diffgrid = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
    print('dt = {}'.format(dt))

    # call the function get_diffgrid()
    diffgrid[i] = get_diffgrid(u_values[i], u_values[-1], dt)


# log-log plot of the grid differences
plt.figure(figsize=(6, 6))
plt.grid(True)
plt.xlabel('$\Delta t$', fontsize=18)
plt.ylabel('$L_1$-norm of the grid differences', fontsize=18)
plt.axis('equal')
plt.loglog(dt_values[: -1], diffgrid[: -1], color='k',
           ls='-', lw=2, marker='o')

r = 2
h = 0.001

dt_values2 = np.array([h, r*h, r**2*h])

u_values2 = np.empty_like(dt_values2, dtype=np.ndarray)

diffgrid2 = np.empty(2)

for i, dt in enumerate(dt_values2):

    N = int(T/dt) + 1   # number of time-steps

    # discretize the time t
    t = np.linspace(0.0, T, N)

    # initialize the array containing the solution for each time-step
    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N-1):
        u[n+1] = euler_step(u[n], f, dt)         # call euler_step()

    # store the value of u related to one grid
    u_values2[i] = u


# calculate f2 - f1
diffgrid2[0] = get_diffgrid(u_values2[1], u_values2[0], dt_values2[1])

# calculate f3 - f2
diffgrid2[1] = get_diffgrid(u_values2[2], u_values2[1], dt_values2[2])

# calculate the orderpand of convergence
p = (log(diffgrid2[1]) - log(diffgrid2[0])) / log(r)

print('The order of convergence is p = {:.3f}'.format(p))
plt.show()

######
