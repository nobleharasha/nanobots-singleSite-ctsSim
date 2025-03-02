# PASSIVE agent model



import numpy as np
from scipy.stats import truncnorm as truncnorm
import math



# Variance calculation for orientation-biased movement model
def var(x, P, D, b, t_):
    phi = np.linalg.norm(x)
    out = (2 * P * phi / (16 * np.pi * D**2 * (t_)**2)) * np.exp((-1.*phi**2) / (4 * D * (t_)))
    if out < 1. / (15. * b):
        return 15.
    else:
        return min(1. / (b * abs(out)), 15.)



# Orientation-biased movement model update step
def ori(x, P, D, b, t_, step_size, phi_boundary):
    mu = -x
    unif = False
    if b=="RW":
        unif = True
    if not unif:
        v = var(x, P, D, b, t_)
        a, bb = -1.*np.pi / v, np.pi / v
        if v >= 15:
            unif = True
    if unif:
        beta = np.random.uniform(-1.*np.pi,np.pi)
    else:
        beta = truncnorm.rvs(loc=0, scale=v, a=a, b=bb, size=1)[0]
    theta = np.array([[np.cos(beta), -1.*np.sin(beta)], [np.sin(beta), np.cos(beta)]]) @ mu.T
    x_new = x + step_size * (theta / np.linalg.norm(theta))

    while np.linalg.norm(x_new) > phi_boundary:
        if unif:
            beta = np.random.uniform(-1.*np.pi,np.pi)
        else:
            beta = truncnorm.rvs(loc=0, scale=v, a=a, b=bb, size=1)[0]
        theta = np.array([[np.cos(beta), -1.*np.sin(beta)], [np.sin(beta), np.cos(beta)]]) @ mu.T
        x_new = x + step_size * (theta / np.linalg.norm(theta))
    return x_new



# Main function to carry out a single simulation run with the given model parameters
def runn(n, phi_0=.005, phi_boundary=.01, b=1e12, P=1e-19, t_=1e4, D=1e-10, step_size=2e-5, epsilon=2e-5, percent_cutoff=.75, time_cutoff=1e7):
    x_0 = np.array([phi_0, 0.])

    drugs = 0.
    t = 0
    xs = [x_0 for _ in range(n)]
    term = [False for _ in range(n)]
    while t < time_cutoff and (drugs/n) < percent_cutoff:
        for i in range(n):
            if term[i]:
                continue
            else:
                xs[i] = ori(xs[i], P, D, b, t_, step_size, phi_boundary)
                if np.linalg.norm(xs[i]) <= epsilon:
                    drugs += 1.
                    term[i] = True
        t += 1

    return t
