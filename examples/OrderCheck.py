import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ODELab.ERK import (ClassicRK4, DormandPrince45,
                        DormandPrince68, BogackiShampine23)
from ODELab.Common import PowerFit



def hooke(t, u):
    return (True, np.array([u[1], -u[0]]))

def exact(t):
    return np.array([np.cos(t), -np.sin(t)])


if __name__=='__main__':

    names = []

    for stepper, name in (
            (BogackiShampine23(),'BS3'),
            (ClassicRK4(), 'RK4'),
            (DormandPrince45(),'DP5'),
            (DormandPrince68(), 'DP8')
            ):

        tInit = 0
        tFinal = 40
        uInit = np.array([1.0, 0.0])

        H = []
        GTE = []
        names.append(name)



        for alpha in range(2,13):
            n = 10*2**alpha
            dt = tFinal/n
            print('%7d %12.5g' % (n, dt))

            uCur = uInit.copy()
            tCur = tInit

            for i in range(n):
                stat, tCur, uCur, err = stepper.step(tCur, uCur, hooke, dt)

            H.append(dt)
            gte = norm(uCur-exact(tFinal))
            GTE.append(gte)
            if gte <= 1.0e-12:
                break

        p = PowerFit(H, GTE)
        print('exponent=', p)

        plt.loglog(H, GTE, 'o-')

    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend(names)
    plt.show()
