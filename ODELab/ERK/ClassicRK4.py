from . ERKStepper import ERKStepper
import numpy as np

class ClassicRK4(ERKStepper):

    def __init__(self):
        c = (0, 1/2, 1/2, 1)
        b = (1/6, 1/3, 1/3, 1/6)
        A = (
            (0,   0,   0,  0),
            (1/2, 0,   0,  0),
            (0,   1/2, 0,  0),
            (0,   0,   1,  0)
            )

        super().__init__(A, b, c, 4)
