import sys
import os
import numpy as np
import gnumpy as gp
import time
import theano
import gc
import ctypes

if sys.platform == 'nt':
    import msvcrt


def steps(start, end, step):
    while start <= end:
        yield start
        start += step


def isfinite(x):
    if not isinstance(x, np.ndarray):
        return x.all_real()
    else:
        return np.all(np.isfinite(x))


def get_key():
    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getch()
        else:
            return None
    else:
        return None


def get_basedir():
    """ Returns base directory of the Addiplication project"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))


def get_datadir():
    """ Returns data directory of the Addiplication project"""
    basedir = get_basedir()
    return os.path.join(basedir, 'data')


def get_randseed():
    return int(time.time())


def cuda_device_reset():
    gc.collect()
    cuda_dll = ctypes.WinDLL("cudart64_55.dll")
    cuda_dll.cudaDeviceReset()


def to_1hot(data, max_value):
    """
    Converts a one-dimensional data sequence to one-hot encoding.
    :param data: integer data array to convert
    :param max_value: maximum value
    :return: onehot[value, smpl]
    """
    n_samples = data.shape[0]
    onehot = np.zeros((max_value + 1, n_samples))
    for smpl, val in enumerate(data):
        val = int(val)
        assert 0 <= val <= max_value
        onehot[val, smpl] = 1.0
    return onehot


class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print '<' * 50
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
            print '>' * 50
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(PrintEverythingMode, self).__init__(
            wrap_linker, optimizer='fast_compile')


class WarnNaNMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            fn()
            for i, inpt in enumerate(fn.inputs):
                try:
                    if np.isnan(inpt[0]).any():
                        print 'nan detected in input %i of %s' % (i, node)
                        import pdb
                        pdb.set_trace()
                except TypeError:
                    print 'could not check for NaN in:', inpt
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(WarnNaNMode, self).__init__(
            wrap_linker, optimizer='fast_compile')


class Output2Floats(object):
    """
    Needed for Theano's scalar ops with two outputs to avoid using
    scalar.upgrade_to_float or upcast_out (which both always return one value)
    """
    def __new__(self, *types):
        type = types[0]
        if type in theano.scalar.float_types:
            return [type, type]
        else:
            return [theano.config.floatX, theano.config.floatX]

