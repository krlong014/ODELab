import matplotlib.pyplot as plt
import copy
import numpy as np

class ArrayOutputHandler:
    def __init__(self, dim, nSteps):
        self.nSteps = nSteps
        self.dim = dim
        self.T = np.zeros((nSteps+1,1))
        self.X = np.zeros((nSteps+1,dim))
        self.V = np.zeros((nSteps+1,dim))

    def acceptStep(self, n, t, x, v):
        self.T[n,0]=t
        self.X[n,:]=x
        self.V[n,:]=v

class VerletDriver:
    def __init__(self, dt, nSteps):
        self.dt = dt
        self.nSteps = nSteps

    def step(self, tCur, xCur, vCur, aCur, model):

        dt = self.dt
        xNew = xCur + dt*(vCur + 0.5*aCur*dt)
        tNew = tCur + dt
        aNew = model.accel(xNew, tNew)
        vNew = vCur + 0.5*dt*(aNew + aCur)

        return (tNew, xNew, vNew, aNew)

    def run(self, txvInit, model, outputHandler):

        t = txvInit[0]
        x = txvInit[1]
        v = txvInit[2]

        outputHandler.acceptStep(0, t, x, v)

        a = model.accel(x, t)

        for n in range(1, self.nSteps+1):
            t, x, v, a = self.step(t, x, v, a, model)
            outputHandler.acceptStep(n, t, x, v)


if __name__=='__main__':

    from SimpleModels import HookeModel
    import matplotlib.pyplot as plt

    hooke = HookeModel(1.0)

    dim = 1
    dt = 0.1
    nSteps = 2000


    outputHandler = ArrayOutputHandler(dim, nSteps)
    verlet = VerletDriver(dt, nSteps)

    x0 = np.array([1.0])
    v0 = np.array([0.0])
    t0 = 0.0

    txvInit = (t0, x0, v0)

    verlet.run(txvInit, hooke, outputHandler)

    X = outputHandler.X
    V = outputHandler.V

    plt.plot(X, V)
    plt.axes().set_aspect('equal')
    plt.show()
