import numpy as np
from numpy.linalg import norm
from . ERKStepper import ERKStepper
from .. Common import StepStatus
from .. Common import ErrorControlParams

class EmbeddedERKStepper(ERKStepper):

    def __init__(self, A, b, c, p, be1, pLow, be2=None,
                params=ErrorControlParams()):
        super().__init__(A, be1, c, p)

        self._params = params

        self._deltaB1 = np.array(be1) - np.array(b)
        if be2 != None:
            self._deltaB2 = np.array(be2) - np.array(b)
        else:
            self._deltaB2 = be2

        self._pLow = pLow

    def pLow(self):
        return self._pLow

    def controlledStep(self, tCur, uCur, func, dtTry):

        params = self._params

        state = StepStatus.TooManyReductions
        dtNew = dtTry
        absUCur = np.abs(uCur)

        for r in range(params.max_reductions):

            stat, tNext, uNext, K = self.basicStep(tCur, uCur, func, dtNew)

            absUNext = np.abs(uNext)
            maxU = np.maximum(np.abs(uCur), absUNext)
            errGoal = params.tol_a + params.tol_r*maxU

            relErr = self.errEstimate(K, dtNew, errGoal)

            fac = (1/relErr)**(1/(self.pLow()+1))
            fac = min(params.fac_max,
                    max(params.fac_min,
                        fac*params.safety))
            dtNew = dtNew * fac

            if relErr <= 1:
                state = StepStatus.GoodStep
                break


        return (state, tNext, uNext, dtNew)



    def errEstimate(self, K, dt, errGoal):

        err1 = np.zeros(len(errGoal))
        if self._deltaB2 != None:
            err2 = np.zeros(len(errGoal))

        for i in range(self.S()):
            err1 += dt*self._deltaB1[i]*K[i,:]
            if self._deltaB2 != None:
                err2 += dt*self._deltaB2[i]*K[i,:]


        relErr = np.amax(np.abs(err1/errGoal))

        if self._deltaB2 != None:
            relErr2 = np.amax(np.abs(err2/errGoal))
            relErr = relErr * relErr/np.hypot(relErr, 0.1*relErr2)

        return relErr
