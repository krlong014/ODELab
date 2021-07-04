from . EmbeddedERKStepper import EmbeddedERKStepper
import numpy as np

class BogackiShampine23(EmbeddedERKStepper):

    def __init__(self, **kwargs):

        c =  (0,     1/2, 3/4, 1)
        b1 = (2/9,  1/3, 4/9, 0)
        b2 = (7/24, 1/4, 1/3, 1/8)
        A = (
                (0,   0,   0,   0),
                (1/2, 0,   0,   0),
                (0,   3/4, 0,   0),
                (2/9, 1/3, 4/9, 0)
            )
        p = 3

        super().__init__(A, b1, b2, c, p, kwargs)
