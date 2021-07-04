from enum import Enum, unique

@unique
class StepStatus(Enum):
    '''
    StepStatus is used to indicate the result of an attempted timestep. 
    '''
    GoodStep = 1
    TooManyReductions = 2
    MinStepReached = 3
    MaxStepReached = 4
    EvalFailed = 5
