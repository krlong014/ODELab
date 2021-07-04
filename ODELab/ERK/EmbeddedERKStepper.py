import numpy as np
from numpy.linalg import norm
from enum import Enum
from . ERKStepper import ERKStepper
from .. Common import StepStatus



class EmbeddedERKStepper(ERKStepper):

    def __init__(self, A, b1, b2, c, p,
                 max_reductions=20,
                 tol=1.0e-8,
                 dtMax=1.0,
                 dtMin=1.0e-4):
        super().__init__(A, b1, c, p)

        self._deltaB = np.array(b2) - np.array(b1)

        self._max_reductions = max_reductions
        self._tol = tol
        self._dtMin = dtMin
        self._dtMax = dtMax

    def step(self, t, u, func, dt):

        n = len(u)
        (state, tNext, uNext, K) = self.basicStep(t, u, func, dt)

        if state == StepStatus.EvalFailed:
            return (state, tNext, uNext)

        err = np.zeros(n)

        for i in range(self.S()):
            err += dt*self._deltaB[i]*K[i,:]

        return (state, tNext, uNext, err)

    def controlledStep(tCur, uCur, func, dtCur):

        state = StepStatus.TooManyReductions
        normU = norm(u)
        dtNew = dtCur

        for r in range(max_reductions):

            uNext, err = self.step(tCur, uCur, func, dtNew)
            normErr = norm(err)

            dtNew = dt*(tol*(1+normU)/normErr)**(1/(self.p()+1))

            if normErr <= tol*normU + tol:
                state = StepStatus.GoodStep

            if np.abs(dtNew) >= dtMax:
                dtNew = dtMax
                state = StepStatus.MaxStepReached
            if np.abs(dtNew) <= dtMin:
                dtNew = dtMin
                state = StepStatus.MinStepReached

            if state != StepStatus.TooManyReductions:
                break

        return (state, uNext, dtNew)
