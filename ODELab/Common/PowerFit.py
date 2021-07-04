import numpy as np
import numpy.linalg as la

def PowerFit(X, Y):
    '''
    Fit a power law y(x)=c[0]*x**c[1] to a set of points
    X=[x_0, x_1, ..., x_{n-1}]
    Y=[y_0, y_1, ..., y_{n-1}]
    where n>=2.
    Take logs to find log(y)=c[0] + c[1]*log(x), and then do linear regression
    to find the best-fit coefficients c[0] and c[1]. The least squares problem
    is solved by forming the normal equations, which in this case is a 2 by 2
    system.

    Input:
    * X -- vector of independent variables
    * Y -- vector of dependent variables

    Output:
    * c -- vector containing the coefficient c[0] and the power c[1].
    '''

    # Form the matrix of basis functions evaluated at the independent variables.
    # A = [ [1,1,...,1], [log(x_0), log(x_1), ..., log(x_{n-1})] ]^T
    A = np.ones((len(X), 2))
    A[:,1] = np.log(X)

    # Form the normal equations A^T A c = A^T log(Y)
    AtA = np.dot(np.transpose(A), A)
    Atb = np.dot(np.transpose(A), np.log(Y))

    # Solve the normal equations
    c = la.solve(AtA, Atb)

    return c
