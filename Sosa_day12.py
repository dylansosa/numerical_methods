# -*- coding: utf-8 -*-
"""
Dylan
Numerical and Scientific Methods
Day 12
Python 3 & Python 2.7 compatible
"""

from math import sin, cos, log, ceil, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

np.set_printoptions(threshold=np.nan)

T = 0.5  # total run time
dt = 0.01  # time increment
N = int(T/dt) + 1  # number of time-steps

u_be = np.empty(N)  # be = Backward Euler
u_euler = np.empty(N)  # euler = Forward Euler
u_rk4 = np.empty(N)  # rk4 = Fourth-order Runge Kutta
u_y = np.empty(N) # implicit
u_be[0] = 1.0  # fill 1st element with init. values
u_euler[0] = 1.0  # fill 1st element with init. values
u_rk4[0] = 1.0  # fill 1st element with init. values
u_y[0] = 1.0


def f(t, u):
    """Returns the right-hand side of the equation (u_dot).

    Parameters
    ----------
    u : float
        array containing the solution at time n.

    Returns
    -------
    u_dot : float
        array containing the RHS given u.
    """

    u_dot = 1 - t + 4.0*u

    return u_dot


def euler_step(u, f, t, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : float
        approximate solution at the next time step.
    """

    return u + dt * f(t, u)


def runge_kutta4(u, f, t, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 :  float
        approximate solution at the next time step.
    """
    k1 = f(t, u)
    k2 = f((t + dt/2.0), (u+dt/2.0*k1))
    k3 = f((t + dt/2.0), (u+dt/2.0*k2))
    k4 = f((t + dt), (u+dt*k3))

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop -  (RK4) method
for n in range(N-1):
    t = n*dt

    u_euler[n+1] = euler_step(u_euler[n], f, t, dt)

    u_rk4[n+1] = runge_kutta4(u_rk4[n], f, t, dt)

    # Implicitly solve for u_be[n+1] (Backward Euler Method)
    #u_be[n+1] = 5_yn+1 - t_n+1
    u_be[n+1] = (u_be[n] + dt - (dt * dt * (n+1)))/(1-4*dt)

    # y is the exact solution
    u_y[n+1] = (n+1)*dt/4.0 - 3.0/16.0 + 19.0/16.0*exp(4.0*(n+1)*dt)

    # Compare values of each method versus the analytic solution.
    # Plot each method on one plot as a function of time
    # Show rates of convergence for each method
        # Couldn't get the convergence code to work...shapes of arrays were different

print "backward euler array:\n", u_be
print "forward euler array:\n", u_euler

###########################################################
# Determine the convergence of the numerical solutions
#
# dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
#
# u_values = np.empty_like(dt_values, dtype=np.ndarray)
#
# for i, dt in enumerate(dt_values):
    # N = int(T/dt) + 1    # number of time-steps
#
    # t = np.linspace(0.0, T, N)
#
    # u = np.empty((N, u_rk4.size)) # The problem is here
    # u[0] = np.array([u_rk4])
#
    # for n in range(N-1):
        # u[n+1] = runge_kutta4(u[n], f, t, dt)   # call euler_step()
#
    # u_values[i] = u
#
#
# def get_diffgrid(u_current, u_fine, dt):
    # """Returns the difference between one grid and the fine one using L-1 norm.
#
    # Parameters
    # ----------
    # u_current : array of float
        # solution on the current grid.
    # u_finest : array of float
        # solution on the fine grid.
    # dt : float
        # time-increment on the current grid.
#
    # Returns
    # -------
    # diffgrid : float
        # difference computed in the L-1 norm.
    # """
#
    # N_current = float(len(u_current[:, 0]))
    # N_fine = float(len(u_fine[:, 0]))
#
    #grid_size_ratio = int(ceil((N_fine-1)/(N_current-1)))
    # grid_size_ratio = int(ceil(N_fine/N_current))
#
    # diffgrid = dt * np.sum(np.abs(u_current[:, 2]
                                  # - u_fine[::grid_size_ratio, 2]))
#
    # return diffgrid
#
#
# diffgrid = np.empty_like(dt_values)
#
# for i, dt in enumerate(dt_values):
    # print('dt = {}'.format(dt))
#
    # diffgrid[i] = get_diffgrid(u_values[i], u_values[-1], dt)
#
#
# plt.figure(figsize=(6, 6))
# plt.grid(True)
# plt.xlabel('$\Delta t$', fontsize=18)
# plt.ylabel('$L_1$-norm of the grid differences', fontsize=18)
# plt.axis('equal')
# plt.loglog(dt_values[: -1], diffgrid[: -1], color='k',
           # ls='-', lw=2, marker='o')

#################################


plt.xlabel(r'Time')
plt.ylabel(r'Y')
plt.title('Day 12')
plt.plot(u_euler)
plt.plot(u_be)
plt.plot(u_rk4)
plt.plot(u_y,'y^')
plt.legend(['Forward Euler','Backward Euler','Runge-Kutte','Exact Solution'])
plt.show()
