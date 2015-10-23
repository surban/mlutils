"""Math routines that work on Numpy and Gnumpy arrays"""

import gpu
import numpy as np
if gpu.GPU:
    import gnumpy as gp
else:
    gp = None

def is_np(array):
    return isinstance(array, (np.ndarray, float, int))

def is_gp(array):
    if gp is None:
        return False
    return isinstance(array, gp.garray)

def check_type(array):
    if not (is_np(array) or is_gp(array)):
        raise TypeError("array must be either a float, int, Numpy array or a Gnumpy array"
                        " (or GPU is not available and a Gnumpy array was passed)")

def abs(x):
    check_type(x)
    if is_np(x):
        return np.abs(x)
    else:
        return gp.abs(x)

def where(x):
    check_type(x)
    if is_np(x):
        return np.where(x)
    else:
        return gp.where(x)

def sum(x, **kwargs):
    check_type(x)
    if is_np(x):
        return np.sum(x, **kwargs)
    else:
        return gp.sum(x, **kwargs)

def sqrt(x):
    check_type(x)
    if is_np(x):
        return np.sqrt(x)
    else:
        return gp.sqrt(x)

def sign(x):
    check_type(x)
    if is_np(x):
        return np.sign(x)
    else:
        return gp.sign(x)




