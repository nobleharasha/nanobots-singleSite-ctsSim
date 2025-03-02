# ACTIVE agent model



import numpy as np
from scipy.stats import truncnorm as truncnorm
import math



# Variance calculation for orientation-biased movement model
def var(x, t, TT, P, D, b):
    phi = np.linalg.norm(x)
    out = 0.
    for t_ in TT:
        if t == t_:
            continue
        out += (phi / (t-t_)**2) * np.exp(-1.*phi**2 / (4*D*(t-t_)))
    out *= P / (8 * np.pi * D**2)
    if out < 1. / (15. * b):
        return 15.
    else:
        return min(1. / (b * abs(out)), 15.)



# Orientation-biased movement model update step
def ori(x, t, TT, P, D, b, step_size, phi_boundary):
    mu = -x
    unif = False
    if b=="RW":
        unif = True
    if not unif:
        v = var(x, t, TT, P, D, b)
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
def runn(n, sigbool=False, phi_0=.005, phi_boundary=.01, b=1e12, P=1e-19, D=1e-9, step_size=2e-5, epsilon=2e-5, percent_cutoff=.75, time_cutoff=1e7):
    x_0 = np.array([phi_0, 0.])

    if sigbool:
        TT = [0]
    else:
        TT = []
    term = [False for _ in range(n)]
    drugged = 0.
    t = 0
    classes = [0 for _ in range(math.floor(n/2))] + [1 for _ in range(math.ceil(n/2))]  #0 = signal chemical payload, 1 = drug payload
    xs = [x_0 for _ in range(n)]
    while t < time_cutoff and (drugged/float(math.ceil(n/2))) < percent_cutoff:
        for i in range(n):
            if term[i]:
                continue
            else:
                xs[i] = ori(xs[i], t, TT, P, D, b, step_size, phi_boundary)
                if np.linalg.norm(xs[i]) <= epsilon:
                    if classes[i] == 0:
                        TT.append(t)
                    else:
                        drugged += 1.
                    term[i] = True
        t += 1

    return t
