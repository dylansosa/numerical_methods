# -*- coding: utf-8 -*-
"""
Dylan Sosa
Numerical and Scientific Methods
Day 11
Python 3 & Python 2.7 compatible
"""

from math import sin, cos, log, ceil, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters:
a_r = 0.2  # birth rate per year of rabbits
b_r = 0.001  # death rate per year of rabbits
a_f = 0.001  # birth rate per year of foxes
b_f = 0.5  # death rate per year of foxes

# set initial conditions
rabbits0 = 1000.0  # Initial number of rabbits
foxes0 = 100.0  # Initial number of foxes

T = 80  # total run time in years
dt = 0.1  # time increment of 0.1 years
N = int(T/dt) + 1  # number of time-steps
t = np.linspace(0, T, N)  # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 2))
u[0] = np.array([rabbits0, foxes0])  # fill 1st element with init. values


def f(u):
    """Returns the right-hand side of the predator-prey system
    of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    rabbits = u[0]
    foxes = u[1]

    u_dot = np.empty_like(u)

    # Update the values of u_dot to give the correct formula
    u_dot[0] = a_r*u[0] - b_r * u[0] * u[1]
    u_dot[1] = a_f * u[0] * u[1] - b_f * u[1]

    return u_dot


def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f(u)  # Note how we are able to call another function f(u)


def runge_kutta4(u, f, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    k1 = f(u)
    k2 = f(u+dt/2.0*k1)
    k3 = f(u+dt/2.0*k2)
    k4 = f(u+dt*k3)

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop - Runge Kutta (RK4) method
for n in range(N-1):
    u[n+1] = runge_kutta4(u[n], f, dt)



# visualization of the populations as a function of time
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'Time (years)', fontsize=18)
plt.ylabel(r'Population', fontsize=18)
plt.ylim([0, 3000])
plt.title('Population versus Time = %.2f' % T, fontsize=18)
plt.plot(t, u[:, 0], '-', lw=2, label='rabbits')
plt.plot(t, u[:, 1], '-', lw=2, label='foxes')
plt.legend()


# set initial conditions
rabbits0_2 = 100.0  # Initial number of rabbits
foxes0_2 = 1000.0  # Initial number of foxes

T = 80  # total run time in years
dt = 0.1  # time increment of 0.1 years
N = int(T/dt) + 1  # number of time-steps
t = np.linspace(0, T, N)  # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 2))
u[0] = np.array([rabbits0_2, foxes0_2])  # fill 1st element with init. values




def f2(u):
    """Returns the right-hand side of the predator-prey system
    of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    rabbits = u[0]
    foxes = u[1]

    u_dot = np.empty_like(u)

    # Update the values of u_dot to give the correct formula
    u_dot[0] = a_r*u[0] - b_r * u[0] * u[1]
    u_dot[1] = a_f * u[0] * u[1] - b_f * u[1]

    return u_dot


def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f2(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f2(u)  # Note how we are able to call another function f2(u)


def runge_kutte4_2(u, f, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f2(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    k1 = f2(u)
    k2 = f2(u+dt/2.0*k1)
    k3 = f2(u+dt/2.0*k2)
    k4 = f2(u+dt*k3)

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop - Runge Kutta (RK4) method
for n in range(N-1):
    u[n+1] = runge_kutte4_2(u[n], f2, dt)



# visualization of the populations as a function of time
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'Time (years)', fontsize=18)
plt.ylabel(r'Population', fontsize=18)
plt.ylim([0, 3000])
plt.title('Population versus Time = %.2f' % T, fontsize=18)
plt.plot(t, u[:, 0], '-', lw=2, label='rabbits')
plt.plot(t, u[:, 1], '-', lw=2, label='foxes')
plt.legend()



# set initial conditions
rabbits0_3 = 520.0  # Initial number of rabbits
foxes0_3 = 180.0  # Initial number of foxes

T = 80  # total run time in years
dt = 0.1  # time increment of 0.1 years
N = int(T/dt) + 1  # number of time-steps
t = np.linspace(0, T, N)  # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 2))
u[0] = np.array([rabbits0_3, foxes0_3])  # fill 1st element with init. values




def f3(u):
    """Returns the right-hand side of the predator-prey system
    of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    rabbits = u[0]
    foxes = u[1]

    u_dot = np.empty_like(u)

    # Update the values of u_dot to give the correct formula
    u_dot[0] = a_r*u[0] - b_r * u[0] * u[1]
    u_dot[1] = a_f * u[0] * u[1] - b_f * u[1]

    return u_dot


def euler_step(u, f2, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f3(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f3(u)  # Note how we are able to call another function f3(u)


def runge_kutte4_3(u, f3, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f3(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    k1 = f3(u)
    k2 = f3(u+dt/2.0*k1)
    k3 = f3(u+dt/2.0*k2)
    k4 = f3(u+dt*k3)

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop - Runge Kutta (RK4) method
for n in range(N-1):
    u[n+1] = runge_kutte4_3(u[n], f3, dt)



# visualization of the populations as a function of time
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'Time (years)', fontsize=18)
plt.ylabel(r'Population', fontsize=18)
plt.ylim([0, 3000])
plt.title('Population versus Time = %.2f' % T, fontsize=18)
plt.plot(t, u[:, 0], '-', lw=2, label='rabbits')
plt.plot(t, u[:, 1], '-', lw=2, label='foxes')
plt.legend()


# set initial conditions
rabbits0_4 = 520.0  # Initial number of rabbits
foxes0_4 = 220.0  # Initial number of foxes

T = 80  # total run time in years
dt = 0.1  # time increment of 0.1 years
N = int(T/dt) + 1  # number of time-steps
t = np.linspace(0, T, N)  # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 2))
u[0] = np.array([rabbits0_4, foxes0_4])  # fill 1st element with init. values




def f4(u):
    """Returns the right-hand side of the predator-prey system
    of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    rabbits = u[0]
    foxes = u[1]

    u_dot = np.empty_like(u)

    # Update the values of u_dot to give the correct formula
    u_dot[0] = a_r*u[0] - b_r * u[0] * u[1]
    u_dot[1] = a_f * u[0] * u[1] - b_f * u[1]

    return u_dot


def euler_step(u, f2, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f4(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f4(u)  # Note how we are able to call another function f4(u)


def runge_kutte4_4(u, f4, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f4(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    k1 = f4(u)
    k2 = f4(u+dt/2.0*k1)
    k3 = f4(u+dt/2.0*k2)
    k4 = f4(u+dt*k3)

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop - Runge Kutta (RK4) method
for n in range(N-1):
    u[n+1] = runge_kutte4_4(u[n], f4, dt)



# visualization of the populations as a function of time
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'Time (years)', fontsize=18)
plt.ylabel(r'Population', fontsize=18)
plt.ylim([0, 3000])
plt.title('Population versus Time = %.2f' % T, fontsize=18)
plt.plot(t, u[:, 0], '-', lw=2, label='rabbits')
plt.plot(t, u[:, 1], '-', lw=2, label='foxes')
plt.legend()


# set initial conditions
rabbits0_5 = 500.0  # Initial number of rabbits
foxes0_5 = 200.0  # Initial number of foxes

T = 80  # total run time in years
dt = 0.1  # time increment of 0.1 years
N = int(T/dt) + 1  # number of time-steps
t = np.linspace(0, T, N)  # time discretization

# initialize the array containing the solution for each time-step
u = np.empty((N, 2))
u[0] = np.array([rabbits0_5, foxes0_5])  # fill 1st element with init. values




def f5(u):
    """Returns the right-hand side of the predator-prey system
    of equations (u_dot).

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    u_dot : array of float
        array containing the RHS given u.
    """

    rabbits = u[0]
    foxes = u[1]

    u_dot = np.empty_like(u)

    # Update the values of u_dot to give the correct formula
    u_dot[0] = a_r*u[0] - b_r * u[0] * u[1]
    u_dot[1] = a_f * u[0] * u[1] - b_f * u[1]

    return u_dot


def euler_step(u, f5, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f5(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f5(u)  # Note how we are able to call another function f5(u)


def runge_kutte4_5(u, f5, dt):
    """Returns the solution at the next time-step using the fourth-order
    accurate Runge Kutta (RK4) method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
        f5(u) = u_dot
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    k1 = f5(u)
    k2 = f5(u+dt/2.0*k1)
    k3 = f5(u+dt/2.0*k2)
    k4 = f5(u+dt*k3)

    return u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


# time loop - Runge Kutta (RK4) method
for n in range(N-1):
    u[n+1] = runge_kutte4_5(u[n], f5, dt)



# visualization of the populations as a function of time
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'Time (years)', fontsize=18)
plt.ylabel(r'Population', fontsize=18)
plt.ylim([0, 3000])
plt.title('Population versus Time = %.2f' % T, fontsize=18)
plt.plot(t, u[:, 0], '-', lw=2, label='rabbits')
plt.plot(t, u[:, 1], '-', lw=2, label='foxes')
plt.legend()
plt.show()
