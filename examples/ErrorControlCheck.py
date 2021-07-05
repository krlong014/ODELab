import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ODELab.ERK import DormandPrince45
from ODELab.Common import ErrorControlParams




def hooke(t, u):
    return (True, np.array([u[1], -u[0]]))

def exact(t):
    return np.array([np.cos(t), -np.sin(t)])


if __name__=='__main__':

    params = ErrorControlParams()
    params.tol_r = 1.0e-8

    stepper = DormandPrince45(params)

    t = 0
    tFinal = 40
    uInit = np.array([1.0, 0.0])
    u = uInit.copy()
    dt = 0.25

    X = [u[0]]
    Y = [u[1]]

    while t < tFinal:

        if dt > tFinal-t:
            dt = tFinal - t

        state, t, u, dt = stepper.controlledStep(t, u, hooke, dt)

        print('t={}, dt={}, u={}'.format(t, dt, u))
        X.append(u[0])
        Y.append(u[1])

    plt.axes().set_aspect(1.0)
    plt.plot(X,Y,'-')
    plt.show()
