import numpy as np
from .. Common import StepStatus


class ERKStepper:
    '''
    A generic explicit Runge-Kutta (ERK) step formula can be described
    completely with the Butcher tableau consisting of a vector b in R^S, a
    vector c in R^S, and a strictly lower triangular S by S matrix A.
    The step from t_n to t_{n+1} with timestep h is then constructed
    as follows:

    Compute the stage vectors
        K_i = f(t_n + c_i h, u_n + h \sum_{j=0}^{i-1} A_{i,j} K_j)
    for i = 0, 1, ..., S-1. In many cases these can be regarded as
    approximations to the derivative of the solution evaluated at
    intermediate times t_i = t_n + h c_i.

    Then compute the step as a weighted average of the stage variables,
    u_{n+1} = u_n + h \sum_{j=0}^{S-1} b_j K_j.
    This can be regarded as approximating (by quadrature) the integral of
    the solution's derivative over the timestep.
    '''

    def __init__(self, A, b, c, p):
        '''
        Construct a generic ERK stepper by passing in the vectors and matrix
        from the Butcher tableau as well as an integer p denoting the order
        of the method.

        This constructor will not normally be called by an end user; rather,
        it will usually be called by a derived class whose contructor sets
        up A, b, and c and then calls this constructor via super(). For an
        example, see ClassicRK4.py.
        '''
        # Butcher tableau
        self._A = np.array(A)
        self._b = np.array(b)
        self._c = np.array(c)

        # Order: defined so that the formal global truncation error estimate
        # is O(h^{p}).
        self._p = p

        # Number of stages
        self._S = len(c)


    def step(self, t, u, func, dt):
        '''Carry out a single step without any error estimation or error
        control.

        Input:
        * t -- the simulation time at the start of the step
        * u -- the solution at the start of the step
        * func -- the RHS of the differential equation, to be called as f(t,u)
        * dt -- the timestep

        Output:
        (stat, tNext, uNext, None), where
        * stat --  StepStatus object providing information about the success
        (or otherwise) of the step.
        * tNext -- the simulation time at the end of the step
        * uNext -- the solution at the end of the step
        * None -- nothing; this is here as a placeholder for
        '''

        # The actual stepping is done by the basicStep() function, which
        # returns the status, time, and solution along with the stage variables
        # K. The
        stat, tNext, uNext, K = self.basicStep(t, u, func, dt)

        # Return the results
        return (stat, tNext, uNext, None)


    def A(self):
        '''Return the matrix A from the Butcher tableau'''
        return self._A

    def b(self):
        '''Return the vector b from the Butcher tableau'''
        return self._b

    def c(self):
        '''Return the vector b from the Butcher tableau'''
        return self._c

    def S(self):
        '''Return the number of stages'''
        return self._S

    def p(self):
        '''Return the order of the method'''
        return self._p

    def basicStep(self, t, u, func, dt):

        n = len(u)

        K = np.zeros((self.S(),n))

        A = self._A
        b = self._b
        c = self._c
        S = self._S


        for i in range(S):
            u_i = np.copy(u)
            for j in range(i):
                if A[i,j] != 0:
                    u_i += dt*A[i,j]*K[j,:]
            evalOK, K[i,:] = func(t+c[i]*dt, u_i)
            if not evalOK:
                return (StepStatus.EvalFailed, t, None, None)

        uNext = np.copy(u)
        for i in range(S):
            if b[i]!=0:
                uNext += dt*b[i]*K[i,:]

        tNext = t + dt

        return (StepStatus.GoodStep, tNext, uNext, K)
