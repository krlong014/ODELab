import numpy

class ErrorControlParams:
    def __init__(self,
                 max_reductions=20,
                 tol_r=1.0e-4,
                 tol_a=1.0e-10,
                 safety=0.8,
                 fac_min=0.3,
                 fac_max=1.5):

        self.max_reductions = max_reductions
        self.tol_r = tol_r
        self.tol_a = tol_a
        self.safety = safety
        self.fac_min = fac_min
        self.fac_max = fac_max
